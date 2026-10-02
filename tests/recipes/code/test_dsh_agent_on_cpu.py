import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from recipes.code.dsh_agent import DshSdkAgent, validate_result

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
