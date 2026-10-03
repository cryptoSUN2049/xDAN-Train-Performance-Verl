import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from recipes.code import dsh_runner
from recipes.code.dsh_agent import _EMPTY_PATCHES_HASH, DshSdkAgent, validate_result

TRACE = b'{"type":"turn/end","data":{"reason":{"kind":"completed"}}}\n'


def result(trace=TRACE):
    return {
        "schema": "dsh.uni-agent.dsh-run.v1",
        "dsh_session_id": "dsh-session-1",
        "trace_path": "/tmp/run/session.jsonl",
        "trace_persisted": True,
        "trace_sha256": "sha256:" + hashlib.sha256(trace).hexdigest(),
        "event_count": 1,
        "finish_reason": "completed",
        "final_response": "fixed",
        "profile": "sdk-minimal",
        "patches_sha256": "sha256:" + hashlib.sha256(b"[]").hexdigest(),
    }


def validate(value, trace=TRACE):
    return validate_result(value, trace, session_id="dsh-session-1", trace_path="/tmp/run/session.jsonl")


def test_valid_completed_result():
    assert validate(result()) == ("Completed", "fixed")


@pytest.mark.parametrize("response", ["", "partial policy output"])
def test_token_budget_exhaustion_is_a_gradable_truncation(response):
    trace = b'{"type":"turn/end","data":{"reason":{"kind":"max-tokens"}}}\n'
    value = result(trace) | {"finish_reason": "max-tokens", "final_response": response}
    assert validate(value, trace) == ("LimitsExceeded", response)


@pytest.mark.parametrize("event_reason", ["completed", "error", "provider-error"])
def test_token_limit_requires_matching_terminal_event(event_reason):
    trace = (json.dumps({"type": "turn/end", "data": {"reason": {"kind": event_reason}}}) + "\n").encode()
    with pytest.raises(RuntimeError):
        validate(result(trace) | {"finish_reason": "max-tokens"}, trace)


def test_token_limit_cannot_claim_completed():
    trace = b'{"type":"turn/end","data":{"reason":{"kind":"max-tokens"}}}\n'
    with pytest.raises(RuntimeError):
        validate(result(trace), trace)


def test_token_limit_without_terminal_event_is_rejected():
    trace = b'{"type":"assistant/message","data":{}}\n'
    with pytest.raises(RuntimeError):
        validate(result(trace) | {"finish_reason": "max-tokens"}, trace)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", "other"),
        ("dsh_session_id", "another-session"),
        ("trace_path", "/tmp/other"),
        ("trace_persisted", False),
        ("trace_sha256", "sha256:bad"),
        ("event_count", 2),
        ("profile", "headless"),
        ("patches_sha256", "sha256:bad"),
        ("finish_reason", None),
        ("finish_reason", "provider-error"),
        ("final_response", {}),
    ],
)
def test_rejects_untrusted_result(key, value):
    with pytest.raises(RuntimeError):
        validate(result() | {key: value})


def test_rejects_changed_trace():
    with pytest.raises(RuntimeError, match="trace"):
        validate(result(), b"{}\n")


def test_result_cannot_claim_completion_without_matching_event():
    trace = b'{"type":"turn/end","data":{"reason":{"kind":"error"}}}\n'
    with pytest.raises(RuntimeError, match="completion"):
        validate(result(trace), trace)


class Sandbox:
    config = SimpleNamespace(cwd="/workspace/repo")

    def __init__(self, *, fail=False):
        self.files = {}
        self.commands = []
        self.fail = fail

    def copy_to(self, src, dest):
        self.files[dest] = Path(src).read_bytes()

    def copy_out(self, src, dest, **kwargs):
        Path(dest).write_bytes(self.files[src])

    def execute(self, command, *, timeout, **kwargs):
        self.commands.append(command)
        if "bootstrap.py" in command:
            root = next(Path(p).parent.as_posix() for p in self.files if p.endswith("/input.json"))
            if self.fail:
                self.files[root + "/run.log"] = b"SDK diagnostic with private-token"
                return {"reason": "pod_timeout", "returncode": 124, "output": ""}
            payload = json.loads(self.files[root + "/input.json"])
            trace = TRACE
            self.files[root + "/session.jsonl"] = trace
            self.files[root + "/result.json"] = json.dumps(
                result(trace) | {"dsh_session_id": payload["session_id"], "trace_path": root + "/session.jsonl"}
            ).encode()
        return {"reason": "ok", "returncode": 0, "output": ""}


def model():
    return SimpleNamespace(
        config=SimpleNamespace(
            model_name="policy",
            model_kwargs={"base_url": "https://gateway.example/sessions/session-1/v1", "api_key": "private-token"},
        )
    )


def test_agent_preserves_gateway_and_keeps_secrets_out_of_commands(tmp_path):
    sandbox = Sandbox()
    agent = DshSdkAgent(model(), sandbox, msg_path=tmp_path / "main.log")
    assert agent.run("fix 'quoted' $(task)") == ("Completed", "fixed")
    env_file = next(v for k, v in sandbox.files.items() if k.endswith("/env.json"))
    settings = json.loads(env_file)
    assert settings["DSH_UA_BASE_URL"] == model().config.model_kwargs["base_url"]
    assert settings["DSH_UA_MODEL"] == "policy"
    assert settings["DSH_UA_CWD"] == "/workspace/repo"
    assert settings["DSH_RUNTIME_MODE"] == "exe"
    assert all("private-token" not in command and "$(task)" not in command for command in sandbox.commands)
    assert (tmp_path / "dsh-session.jsonl").is_file()


def test_agent_timeout_is_infra_failure():
    with pytest.raises(RuntimeError, match="pod_timeout"):
        DshSdkAgent(model(), Sandbox(fail=True)).run("fix it")


def test_explicit_context_window_reaches_real_helper_environment():
    sandbox = Sandbox()
    agent = DshSdkAgent(model(), sandbox, context_window=65536, run_timeout=1800)
    assert agent.run("fix it") == ("Completed", "fixed")
    settings = json.loads(next(value for key, value in sandbox.files.items() if key.endswith("/env.json")))
    assert settings["DSH_CONTEXT_WINDOW"] == "65536"
    assert settings["DSH_UA_MAX_TOKENS"] == "4096"


@pytest.mark.parametrize("context_window", [0, -1, True, "65536", 65536.0])
def test_invalid_context_window_fails_before_sandbox_execution(context_window):
    sandbox = Sandbox()
    with pytest.raises(ValueError, match="context_window"):
        DshSdkAgent(model(), sandbox, context_window=context_window)
    assert not sandbox.commands


def test_unspecified_context_preserves_existing_sdk_profile_defaults():
    sandbox = Sandbox()
    DshSdkAgent(model(), sandbox).run("fix it")
    settings = json.loads(next(value for key, value in sandbox.files.items() if key.endswith("/env.json")))
    assert "DSH_CONTEXT_WINDOW" not in settings


def test_failure_keeps_private_diagnostics_before_sandbox_cleanup(tmp_path):
    with pytest.raises(RuntimeError, match="pod_timeout") as failure:
        DshSdkAgent(model(), Sandbox(fail=True), msg_path=tmp_path / "main.log").run("fix it")
    assert "private-token" not in str(failure.value)
    path = tmp_path / "dsh-failure-run.log"
    assert path.read_bytes() == b"SDK diagnostic with private-token"
    assert path.stat().st_mode & 0o777 == 0o600


def test_agent_rejects_non_session_gateway():
    policy = model()
    policy.config.model_kwargs["base_url"] = "https://commercial.example/v1"
    with pytest.raises(ValueError, match="session"):
        DshSdkAgent(policy, Sandbox()).run("fix it")


class BareSandbox(Sandbox):
    """A task image without /opt/dsh: the runtime appears only after the payload is unpacked."""

    def __init__(self):
        super().__init__()
        self.runtime_installed = False

    def copy_to(self, src, dest, **kwargs):
        super().copy_to(src, dest)

    def execute(self, command, *, timeout, **kwargs):
        if command.startswith("test -x "):
            self.commands.append(command)
            return {"reason": "ok", "returncode": 0 if self.runtime_installed else 1, "output": ""}
        if command.startswith("tar -xzf /tmp/dsh-runtime-"):
            self.commands.append(command)
            self.runtime_installed = True
            return {"reason": "ok", "returncode": 0, "output": ""}
        return super().execute(command, timeout=timeout, **kwargs)


def _payload(tmp_path):
    path = tmp_path / "dsh-runtime.tar.gz"
    path.write_bytes(b"portable dsh prefix")
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_payload_is_injected_when_the_task_image_lacks_dsh(tmp_path):
    sandbox = BareSandbox()
    path, digest = _payload(tmp_path)
    agent = DshSdkAgent(model(), sandbox, payload_path=path, payload_sha256=digest)
    assert agent.run("fix it") == ("Completed", "fixed")
    assert any(dest.startswith("/tmp/dsh-runtime-") for dest in sandbox.files)
    assert any(command.startswith("tar -xzf /tmp/dsh-runtime-") for command in sandbox.commands)


def test_baked_runtime_skips_the_payload(tmp_path):
    sandbox = Sandbox()
    path, digest = _payload(tmp_path)
    DshSdkAgent(model(), sandbox, payload_path=path, payload_sha256=digest).run("fix it")
    assert not any(dest.startswith("/tmp/dsh-runtime-") for dest in sandbox.files)


def test_missing_runtime_without_payload_fails_closed():
    with pytest.raises(RuntimeError, match="no payload_path"):
        DshSdkAgent(model(), BareSandbox()).run("fix it")


def test_payload_hash_mismatch_is_rejected_before_upload(tmp_path):
    sandbox = BareSandbox()
    path, _digest = _payload(tmp_path)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        DshSdkAgent(model(), sandbox, payload_path=path, payload_sha256="0" * 64).run("fix it")
    assert not any(dest.startswith("/tmp/dsh-runtime-") for dest in sandbox.files)


def test_payload_path_and_hash_must_come_together(tmp_path):
    path, _digest = _payload(tmp_path)
    with pytest.raises(ValueError, match="together"):
        DshSdkAgent(model(), Sandbox(), payload_path=path)


class PatchSandbox(Sandbox):
    """Emulates the runner side: decode DSH_UA_PATCHES and hash the uploaded patch bytes."""

    def execute(self, command, *, timeout, **kwargs):
        response = super().execute(command, timeout=timeout, **kwargs)
        if "bootstrap.py" in command:
            root = next(Path(p).parent.as_posix() for p in self.files if p.endswith("/input.json"))
            settings = json.loads(self.files[root + "/env.json"])
            with mock.patch.dict(os.environ, {"DSH_UA_PATCHES": settings["DSH_UA_PATCHES"]}):
                self.patches = dsh_runner._patches_from_env()
            reported = json.loads(self.files[root + "/result.json"])
            reported["patches_sha256"] = dsh_runner._patches_digest([self.files[p] for p in self.patches])
            self.files[root + "/result.json"] = json.dumps(reported).encode()
        return response


def _patch(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def _settings(sandbox):
    return json.loads(next(value for key, value in sandbox.files.items() if key.endswith("/env.json")))


def test_no_patches_keeps_env_and_identity_unchanged():
    sandbox = PatchSandbox()
    agent = DshSdkAgent(model(), sandbox, patches=[])
    assert agent.patches_sha256 == _EMPTY_PATCHES_HASH == "sha256:" + hashlib.sha256(b"[]").hexdigest()
    assert agent.run("fix it") == ("Completed", "fixed")
    assert _settings(sandbox)["DSH_UA_PATCHES"] == "[]"
    assert not any("/patch-" in dest for dest in sandbox.files)
    assert DshSdkAgent(model(), Sandbox()).patches_sha256 == _EMPTY_PATCHES_HASH


def test_patches_are_uploaded_in_order_and_round_trip_through_the_runner(tmp_path):
    first = _patch(tmp_path, "b.yml", "- insert: [{id: one, name: x}]\n")
    second = _patch(tmp_path, "a.yaml", "- insert: [{id: two, name: y}]\n")
    sandbox = PatchSandbox()
    agent = DshSdkAgent(model(), sandbox, patches=[first, second])
    assert agent.run("fix it") == ("Completed", "fixed")
    remote = json.loads(_settings(sandbox)["DSH_UA_PATCHES"])
    root = remote[0].rsplit("/", 1)[0]
    assert remote == [root + "/patch-00.yml", root + "/patch-01.yml"]
    assert root.startswith("/tmp/mimo-dsh-sdk-") and root + "/env.json" in sandbox.files
    assert [sandbox.files[path] for path in remote] == [Path(first).read_bytes(), Path(second).read_bytes()]
    assert sandbox.patches == tuple(remote)


def test_patch_hash_is_deterministic_and_content_sensitive(tmp_path):
    first = _patch(tmp_path, "one.yml", "- insert: []\n")
    second = _patch(tmp_path, "two.yml", "- id: llm\n")
    digest = DshSdkAgent(model(), Sandbox(), patches=[first, second]).patches_sha256
    assert digest == DshSdkAgent(model(), Sandbox(), patches=[first, second]).patches_sha256
    assert digest != DshSdkAgent(model(), Sandbox(), patches=[second, first]).patches_sha256
    assert digest != _EMPTY_PATCHES_HASH
    Path(second).write_text("- id: llm-changed\n")
    assert digest != DshSdkAgent(model(), Sandbox(), patches=[first, second]).patches_sha256


def test_real_runner_hashes_the_patch_files_it_passes_to_the_sdk(monkeypatch, tmp_path):
    paths = [_patch(tmp_path, "patch-00.yml", "- insert: []\n"), _patch(tmp_path, "patch-01.yml", "- id: llm\n")]
    seen = {}

    class Harness:
        def __init__(self, config):
            seen.update(config)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def run(self, prompt, *, session_id):
            return SimpleNamespace(session_id=session_id, events=[{}], finish_reason="completed", final_response="ok")

    sdk = SimpleNamespace(DeepSeekHarness=Harness, DeepSeekHarnessConfig=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "deepseek_harness", sdk)
    monkeypatch.setenv("DSH_UA_PATCHES", json.dumps(paths))
    monkeypatch.setenv("DSH_UA_TRACE_PATH", str(tmp_path / "t.jsonl"))
    monkeypatch.setenv("DSH_UA_KEEP_TRACE", "1")
    for name in ("DSH_UA_MODEL", "DSH_UA_PROVIDER", "DSH_UA_CWD", "DSH_UA_HOME", "DSH_UA_BASE_URL", "DSH_UA_API_KEY"):
        monkeypatch.setenv(name, "x")
    (tmp_path / "input.json").write_text(json.dumps({"prompt": "p", "session_id": "s"}))
    output = dsh_runner.run(tmp_path / "input.json", tmp_path / "result.json")
    assert seen["patches"] == tuple(paths)
    assert output["patches_sha256"] == DshSdkAgent(model(), Sandbox(), patches=paths).patches_sha256


def test_runner_hash_mismatch_fails_closed(tmp_path):
    path = _patch(tmp_path, "p.yml", "- insert: []\n")
    with pytest.raises(RuntimeError, match="patches_sha256"):
        DshSdkAgent(model(), Sandbox(), patches=[path]).run("fix it")  # this fake runner reports the empty stack


def test_repo_relative_patch_matches_the_shipped_profile(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    repo = Path(__file__).resolve().parents[3]
    base = yaml.safe_load((repo / "config/agent/mixed/dsh-sdk-modal.yaml").read_text())
    compact = yaml.safe_load((repo / "config/agent/mixed/dsh-sdk-modal-compact.yaml").read_text())
    patches = compact["agent"].pop("patches")
    assert compact == base and patches == ["config/agent/dsh/patches/compaction.patch.yml"]
    agent = DshSdkAgent(model(), Sandbox(), patches=patches)
    assert agent.patches == [(repo / patches[0]).read_bytes()]
    rows = yaml.safe_load(agent.patches[0])[0]["insert"]
    assert [row["id"] for row in rows] == ["token-meter", "tool-result-pruner", "compaction-basic"]


def test_expected_patch_hash_is_enforced(tmp_path):
    path = _patch(tmp_path, "p.yml", "- insert: []\n")
    digest = DshSdkAgent(model(), Sandbox(), patches=[path]).patches_sha256
    DshSdkAgent(model(), Sandbox(), patches=[path], patches_sha256=digest)
    DshSdkAgent(model(), Sandbox(), patches_sha256=_EMPTY_PATCHES_HASH)
    with pytest.raises(ValueError, match="patches_sha256 mismatch"):
        DshSdkAgent(model(), Sandbox(), patches=[path], patches_sha256=_EMPTY_PATCHES_HASH)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda tmp: [str(tmp / "missing.yml")], "does not exist"),
        (lambda tmp: [_patch(tmp, "p.yml", "[]\n")] * 2, "repeat"),
        (lambda tmp: [_patch(tmp, "p.json", "[]\n")], r"\.yml or \.yaml"),
        (lambda tmp: "config/agent/dsh/patches/compaction.patch.yml", "list of paths"),
        (lambda tmp: [""], "list of paths"),
    ],
)
def test_invalid_patches_fail_before_sandbox_execution(tmp_path, make, match):
    sandbox = Sandbox()
    with pytest.raises(ValueError, match=match):
        DshSdkAgent(model(), sandbox, patches=make(tmp_path))
    assert not sandbox.commands and not sandbox.files
