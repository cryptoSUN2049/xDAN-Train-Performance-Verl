"""Run the pinned xDAN DSH SDK inside an existing MiMo task environment.

The Gateway remains the sole source of training tokens. SDK logs are evidence,
not reconstructed model trajectories. Installation belongs to the image layer.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from recipes.code.dsh_runner import _patches_digest

_EMPTY_PATCHES_HASH = "sha256:" + hashlib.sha256(b"[]").hexdigest()
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP = """import hashlib, importlib.metadata, json, os, pathlib, runpy, sys
root = pathlib.Path(sys.argv[1])
settings = json.loads((root / "env.json").read_text())
os.environ.update(settings)
try:
    for package in ("deepseek-harness-sdk", "deepseek-harness-runtime-bin"):
        if importlib.metadata.version(package) != "0.1.3a2":
            raise RuntimeError("DSH runtime version mismatch")
    from deepseek_harness_runtime import resolve_bundled_launch_args
    binary = pathlib.Path(resolve_bundled_launch_args("exe")[0])
    identities = {
        binary: "d1a467a9c14a38ad5f01591d2cdb125852cb1a1d3b0ecb678dfde383404e80cb",
        pathlib.Path(str(binary) + "-rg"): "193906679498de4d939345b937fa24e0e69a03c244bd70c859f5e41232713f21",
    }
    for path, expected in identities.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("DSH runtime identity mismatch")
    sys.argv = [str(root / "runner.py"), "--input", str(root / "input.json"), "--output", str(root / "result.json")]
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    (root / "env.json").unlink(missing_ok=True)
    (root / "input.json").unlink(missing_ok=True)
"""


def _read_patches(patches: list[str]) -> list[bytes]:
    """Read ordered local DSH profile patches; relative paths resolve from the repo, not the cwd."""
    if not isinstance(patches, list | tuple) or any(not isinstance(item, str) or not item.strip() for item in patches):
        raise ValueError("DSH patches must be a list of paths")
    paths = [(_REPO_ROOT / item).resolve() for item in patches]
    if len(set(paths)) != len(paths):
        raise ValueError("DSH patches must not repeat a file")
    for path in paths:
        if path.suffix not in (".yml", ".yaml"):
            raise ValueError(f"DSH patch {path} must be a .yml or .yaml file")
        if not path.is_file():
            raise ValueError(f"DSH patch {path} does not exist")
    return [path.read_bytes() for path in paths]


def validate_result(
    value: dict, trace: bytes, *, session_id: str, trace_path: str, patches_sha256: str = _EMPTY_PATCHES_HASH
) -> tuple[str, str]:
    """Fail closed on a corrupt, mismatched or incomplete SDK result."""
    expected = {
        "schema": "dsh.uni-agent.dsh-run.v1",
        "dsh_session_id": session_id,
        "trace_path": trace_path,
        "trace_persisted": True,
        "trace_sha256": "sha256:" + hashlib.sha256(trace).hexdigest(),
        "profile": "sdk-minimal",
        "patches_sha256": patches_sha256,
    }
    if not isinstance(value, dict):
        raise RuntimeError("DSH result must be an object")
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise RuntimeError(f"DSH result {key} mismatch")
    try:
        events = [json.loads(line) for line in trace.splitlines() if line.strip()]
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("DSH trace is not JSONL") from exc
    if not events or any(not isinstance(event, dict) for event in events):
        raise RuntimeError("DSH trace must contain event objects")
    if type(value.get("event_count")) is not int or value["event_count"] != len(events):
        raise RuntimeError("DSH trace event_count mismatch")
    finish_reason = value.get("finish_reason")
    if finish_reason not in ("completed", "max-tokens"):
        raise RuntimeError("DSH did not complete normally; inspect the private trace")
    endings = [event for event in events if event.get("type") == "turn/end"]
    reason = (endings[-1].get("data") or {}).get("reason") if endings else None
    if not isinstance(reason, dict) or reason.get("kind") != finish_reason:
        raise RuntimeError("DSH trace has no matching completion/limit event")
    if not isinstance(value.get("final_response"), str):
        raise RuntimeError("DSH final_response must be a string")
    # A verified policy token limit is a gradable truncated attempt. The
    # runner scores the existing task filesystem and marks it incomplete;
    # transport/provider/runtime failures still raise above or in _execute.
    status = "Completed" if finish_reason == "completed" else "LimitsExceeded"
    return status, value["final_response"]


class DshSdkAgent:
    """MiMo Agent protocol, using the same task filesystem and policy session."""

    IDLE_STATUS = "Completed"

    def __init__(
        self,
        model,
        env,
        *,
        python_path: str = "/opt/dsh/bin/python",
        run_timeout: int = 600,
        max_tokens: int = 4096,
        context_window: int | None = None,
        profile: str = "sdk-minimal",
        msg_path: Path | None = None,
        payload_path: str | None = None,
        payload_sha256: str | None = None,
        patches: list[str] | None = None,
        patches_sha256: str | None = None,
    ):
        if profile != "sdk-minimal":
            raise ValueError("Only the frozen sdk-minimal profile is supported")
        if run_timeout <= 0 or max_tokens <= 0 or not Path(python_path).is_absolute():
            raise ValueError("DSH needs positive budgets and an absolute interpreter path")
        if context_window is not None and (type(context_window) is not int or context_window <= 0):
            raise ValueError("DSH context_window must be a positive integer")
        if (payload_path is None) != (payload_sha256 is None):
            raise ValueError("DSH payload_path and payload_sha256 must be set together")
        if payload_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", payload_sha256):
            raise ValueError("DSH payload_sha256 must be a hex SHA256")
        self.model, self.env = model, env
        self.python_path, self.run_timeout, self.max_tokens = python_path, run_timeout, max_tokens
        self.context_window = context_window
        self.msg_path = Path(msg_path) if msg_path else None
        self.payload_path, self.payload_sha256 = payload_path, payload_sha256
        # Snapshot patch contents once: the bytes hashed here are the bytes uploaded,
        # and the runner re-hashes what it reads in the sandbox (validate_result).
        self.patches = _read_patches([] if patches is None else patches)
        self.patches_sha256 = _patches_digest(self.patches)
        if patches_sha256 is not None and patches_sha256 != self.patches_sha256:
            raise ValueError("DSH patches_sha256 mismatch")
        self.messages = []

    def _ensure_runtime(self) -> None:
        """Inject the task-independent /opt/dsh payload when the task image does not bake it in.

        Images built with DSH keep working unchanged. The payload is hash-checked before upload, and
        the bootstrap re-verifies package versions and runtime binary identities inside the sandbox.
        """
        present = self.env.execute(f"test -x {shlex.quote(self.python_path)}", timeout=30)
        if present.get("reason", "ok") == "ok" and present.get("returncode") == 0:
            return
        if self.payload_path is None:
            raise RuntimeError("DSH runtime is not in the task image and no payload_path is configured")
        digest = hashlib.sha256()
        with open(self.payload_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.payload_sha256:
            raise RuntimeError("DSH payload SHA256 mismatch")
        remote = f"/tmp/dsh-runtime-{uuid.uuid4().hex}.tar.gz"
        self.env.copy_to(self.payload_path, remote, timeout=900)
        self._execute(f"tar -xzf {remote} -C /opt && rm -f {remote} && test -x {shlex.quote(self.python_path)}", 600)

    def get_model_query_kwargs(self):
        return {}

    def _execute(self, command: str, timeout: int):
        result = self.env.execute(command, timeout=timeout)
        if result.get("reason", "ok") != "ok" or result.get("returncode") != 0:
            # Output may include authentication values or task content. Keep it
            # in sandbox artifacts rather than embedding it in trainer errors.
            raise RuntimeError(f"DSH sandbox command failed: {result.get('reason')} rc={result.get('returncode')}")

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        model_kwargs = self.model.config.model_kwargs
        base_url = model_kwargs.get("base_url", "").rstrip("/")
        parsed = urlsplit(base_url)
        match = re.fullmatch(r"/sessions/([A-Za-z0-9_-]+)/v1", parsed.path)
        if not match or parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("DSH requires a bound Gateway session URL")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("DSH session URL cannot contain credentials, query or fragment")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("DSH requires a non-empty task")
        session_id = "dsh-" + match[1]
        root = f"/tmp/mimo-dsh-sdk-{uuid.uuid4().hex}"
        trace_path = root + "/session.jsonl"
        patch_files = {f"patch-{index:02d}.yml": content for index, content in enumerate(self.patches)}
        settings = {
            "DSH_RUNTIME_MODE": "exe",
            "DSH_UA_BASE_URL": base_url,
            "DSH_UA_API_KEY": model_kwargs.get("api_key") or "not-needed",
            "DSH_UA_MODEL": self.model.config.model_name,
            "DSH_UA_PROVIDER": "deepseek-official",
            "DSH_UA_CWD": self.env.config.cwd,
            "DSH_UA_HOME": root + "/home",
            "DSH_UA_TRACE_PATH": trace_path,
            "DSH_UA_KEEP_TRACE": "1",
            "DSH_UA_PROFILE": "sdk-minimal",
            "DSH_UA_PATCHES": json.dumps([root + "/" + name for name in patch_files]),
            "DSH_UA_MAX_TOKENS": str(self.max_tokens),
            "DSH_TELEMETRY_DISABLED": "1",
        }
        if self.context_window is not None:
            settings["DSH_CONTEXT_WINDOW"] = str(self.context_window)
        self._ensure_runtime()
        self._execute(f"umask 077; mkdir -m 700 {shlex.quote(root)}", 30)
        with tempfile.TemporaryDirectory(prefix="mimo-dsh-") as directory:
            local = Path(directory)
            files = {
                "input.json": json.dumps({"prompt": task, "session_id": session_id}).encode(),
                "env.json": json.dumps(settings).encode(),
                "runner.py": Path(__file__).with_name("dsh_runner.py").read_bytes(),
                "bootstrap.py": _BOOTSTRAP.encode(),
                **patch_files,
            }
            try:
                for name, content in files.items():
                    path = local / name
                    path.write_bytes(content)
                    path.chmod(0o600)
                    self.env.copy_to(str(path), root + "/" + name)
                command = shlex.join(
                    ["timeout", "-k", "10", str(self.run_timeout), self.python_path, root + "/bootstrap.py", root]
                )
                self._execute(command + f" > {root}/run.log 2>&1", self.run_timeout + 20)
                for name in ("result.json", "session.jsonl"):
                    self.env.copy_out(root + "/" + name, str(local / name))
                trace = (local / "session.jsonl").read_bytes()
                if self.msg_path:
                    self.msg_path.parent.mkdir(parents=True, exist_ok=True)
                    artifact = self.msg_path.with_name("dsh-session.jsonl")
                    artifact.write_bytes(trace)
                    artifact.chmod(0o600)
                status, response = validate_result(
                    json.loads((local / "result.json").read_text()),
                    trace,
                    session_id=session_id,
                    trace_path=trace_path,
                    patches_sha256=self.patches_sha256,
                )
                self.messages = [{"role": "user", "content": task}, {"role": "assistant", "content": response}]
                return status, response
            except Exception:
                if self.msg_path:
                    self.msg_path.parent.mkdir(parents=True, exist_ok=True)
                    for name in ("run.log", "session.jsonl", "result.json"):
                        try:
                            # Retrieve into the private temporary directory first;
                            # credentials in an SDK traceback must never be public.
                            path = local / ("failure-" + name)
                            self.env.copy_out(root + "/" + name, str(path), timeout=15, max_retries=1)
                            artifact = self.msg_path.with_name("dsh-failure-" + name)
                            artifact.touch(mode=0o600)
                            artifact.chmod(0o600)
                            artifact.write_bytes(path.read_bytes())
                        except Exception:
                            pass  # Preserve the primary error if the sandbox is gone.
                raise
            finally:
                # The runner's outer finally destroys the sandbox, including
                # subprocesses, even if transport failure prevents this cleanup.
                try:
                    self.env.execute(f"rm -f {root}/env.json {root}/input.json", timeout=10)
                except Exception:
                    pass
