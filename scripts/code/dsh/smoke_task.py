"""Bounded real SDK / Modal / MiMo grader smoke with a scripted model backend.

This is protocol evidence, never a policy rollout or a training sample. Run on
an isolated cloud CPU controller with Modal credentials already configured.
Raw requests, traces and grader output stay in the private output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from fastapi import Request

IMAGE = (
    "ghcr.io/cryptosun2049/mimo-dsh-code-001661@sha256:15f588d627ce06e17d2904193c07100aa0b73c883269d70565ab59f36f6f2952"
)
SMOKE_PROMPT = (
    "Protocol acceptance only: read /etc/os-release using the command tool, then stop. Do not modify task files."
)


def load_instance(path):
    value = json.loads(Path(path).read_text())
    instance = json.loads(value["extra_info"]["instance_json"])
    if instance.get("instance_id") != "format-code-task-001661":
        raise ValueError("This frozen smoke image only matches format-code-task-001661")
    return {**instance, "docker_image": IMAGE, "verifier_timeout_sec": 180}


def classify_verdict(value, instance):
    """Distinguish this frozen task's missing required API from infrastructure."""
    code = value.get("verifier_returncode")
    if code in {0, 1}:
        return "tests_passed" if code == 0 else "tests_failed"
    if (
        code == 2
        and instance.get("instance_id") == "format-code-task-001661"
        and "friend_set_for" in instance.get("problem_statement", "")
        and "cannot import name 'friend_set_for' from 'friends.models'" in value.get("test_output", "")
    ):
        return "required_task_api_not_implemented"
    raise RuntimeError("Verifier did not produce a recognized task verdict; inspect private output")


class ScriptedPolicy:
    """Issue one real tool call and require its nonce on the next model request."""

    def __init__(self, nonce):
        self.nonce = nonce
        self.calls = 0
        self.tool_result_verified = False
        self.tool_name = None
        self.first_request_sizes = {}

    def respond(self, request):
        self.calls += 1
        if self.calls == 1:
            self.first_request_sizes = {
                name + "_utf8_bytes": len(json.dumps(request.get(name, []), ensure_ascii=False).encode())
                for name in ("messages", "tools")
            }
            self.first_request_sizes["tool_count"] = len(request.get("tools", []))
            functions = [item.get("function", {}) for item in request.get("tools", [])]
            tool = next(
                (
                    item
                    for item in functions
                    if item.get("name") == "bash" and "command" in item.get("parameters", {}).get("properties", {})
                ),
                None,
            )
            if tool is None:
                raise ValueError("SDK did not offer a command tool")
            self.tool_name = tool["name"]
            arguments = {"command": f"cat /etc/os-release >/dev/null && printf '%s\\n' '{self.nonce}'"}
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_smoke_1",
                        "type": "function",
                        "function": {"name": self.tool_name, "arguments": json.dumps(arguments)},
                    }
                ],
            }, "tool_calls"
        results = [item.get("content", "") for item in request.get("messages", []) if item.get("role") == "tool"]
        if not any(isinstance(value, str) and self.nonce in value.splitlines() for value in results):
            raise ValueError("SDK did not return the expected tool result")
        self.tool_result_verified = True
        return {"role": "assistant", "content": "Protocol smoke complete. No task repair was attempted."}, "stop"


def _free_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


def measure_prompt_tokens(request_path, tokenizer_path, instance):
    """Count full templates locally, including tools; never download weights."""
    from transformers import AutoTokenizer

    request = json.loads(request_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    messages = request["messages"]

    def count(value):
        encoded = tokenizer.apply_chat_template(
            value, tools=request.get("tools"), tokenize=True, add_generation_prompt=True, return_dict=False
        )
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)

    result = {"scripted_prompt_tokens": count(messages), "completion_budget": 4096}
    replaced = json.loads(json.dumps(messages))
    changes = 0
    for message in replaced:
        if (
            message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and SMOKE_PROMPT in message["content"]
        ):
            message["content"] = message["content"].replace(SMOKE_PROMPT, instance["problem_statement"])
            changes += 1
    if changes != 1:
        raise ValueError("Cannot reconstruct exactly one original task prompt from SDK messages")
    result["original_task_prompt_tokens"] = count(replaced)
    result["original_task_total_with_completion"] = result["original_task_prompt_tokens"] + 4096
    result["fits_8192_first_turn_only"] = result["original_task_total_with_completion"] <= 8192
    result["scope"] = "first-turn-chat-template-with-tools; excludes-future-tool-results"
    return result


def _serve(app, port):
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("Local smoke server did not start")
        time.sleep(0.05)
    return server, thread


def _backend(policy, private_dir, rewards):
    from fastapi import FastAPI
    from starlette.responses import JSONResponse, StreamingResponse

    app = FastAPI()

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def completions(session_id: str, request: Request):
        value = await request.json()
        path = private_dir / f"request-{policy.calls + 1}.json"
        path.write_text(json.dumps(value))
        path.chmod(0o600)
        try:
            message, reason = policy.respond(value)
        except ValueError:
            return JSONResponse({"error": "scripted backend contract failed"}, status_code=400)
        base = {"id": "chatcmpl-protocol-smoke", "created": int(time.time()), "model": "policy"}
        if not value.get("stream"):
            return {
                **base,
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            }
        delta = dict(message)
        if "tool_calls" in delta:
            delta["tool_calls"] = [{"index": index, **item} for index, item in enumerate(delta["tool_calls"])]
        chunks = [
            {
                **base,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                **base,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            },
        ]
        payload = "".join("data: " + json.dumps(item) + "\n\n" for item in chunks) + "data: [DONE]\n\n"
        return StreamingResponse(iter([payload]), media_type="text/event-stream")

    @app.post("/sessions/{session_id}/reward")
    async def reward(session_id: str, request: Request):
        rewards.append(await request.json())
        artifact = private_dir / "rewards.json"
        artifact.write_text(json.dumps(rewards))
        artifact.chmod(0o600)
        return {"ok": True}

    return app


def run(args):
    import modal
    import yaml
    from mimoagent.environments.modal import ModalEnvironment

    from recipes.code.dsh_gateway_proxy import create_app
    from recipes.code.mimoagent_runner import mimoagent_runner

    os.umask(0o077)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    private = output / "private"
    private.mkdir(mode=0o700)
    route_dir = private / "routes"
    policy = ScriptedPolicy("train-p0-tool-" + secrets.token_hex(12))
    rewards = []
    servers = []
    tunnel = None
    instances = []
    evidence = {
        "schema": "train-p0.dsh-task-smoke.v1",
        "scope": "scripted-backend-protocol-only",
        "status": "starting",
        "image": IMAGE,
        "cpu": 1,
        "memory_mib": 4096,
        "gpu": None,
        "policy_model_used": False,
        "training_samples": 0,
        "task_id": "format-code-task-001661",
        "verifier_timeout_override_seconds": 180,
        "sandbox_timeout_seconds": 600,
        "git_leak_prevention": "strip",
        "anti_hack_cleanup": True,
        "source_row_sha256": hashlib.sha256(Path(args.row).read_bytes()).hexdigest(),
    }
    root = Path(__file__).resolve().parents[3]
    evidence["source_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "recipes/code/mimoagent_runner.py",
            "recipes/code/code_environment.py",
            "recipes/code/dsh_agent.py",
            "recipes/code/dsh_runner.py",
            "recipes/code/dsh_gateway_proxy.py",
            "scripts/code/dsh/smoke_task.py",
        )
    }
    started = time.monotonic()

    def save():
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(
            json.dumps({k: evidence[k] for k in ("status", "stage", "sandbox_id", "terminated") if k in evidence}),
            flush=True,
        )

    class ObservedModal(ModalEnvironment):
        def start(self):
            instances.append(self)
            super().start()
            evidence["sandbox_id"] = self.sandbox_id
            # Hidden test contents must not exist before the agent executes.
            result = self.execute(
                "test ! -e mimo_test_command.sh && test ! -e usercase-test-coderl/test_bidirectional_friendship.py"
            )
            evidence["hidden_tests_absent_before_agent"] = result["returncode"] == 0
            if not evidence["hidden_tests_absent_before_agent"]:
                raise RuntimeError("Hidden tests exposed before rollout")
            save()

        def execute(self, command, *positional, **kwargs):
            result = super().execute(command, *positional, **kwargs)
            if "/bootstrap.py" in command and result.get("returncode") != 0:
                match = re.search(r"(/tmp/mimo-dsh-sdk-[a-f0-9]+)/bootstrap.py", command)
                if match:
                    self.copy_out(match[1] + "/run.log", str(private / "dsh-failure.log"))
            return result

        def cleanup(self):
            sandbox_id = self.sandbox_id
            super().cleanup()
            if sandbox_id:
                remote = modal.Sandbox.from_id(sandbox_id)
                code = remote.poll()
                if code is None:
                    remote.terminate(wait=True)
                    code = modal.Sandbox.from_id(sandbox_id).poll()
                evidence.update(independent_exit_code=code, terminated=code is not None)
                save()

    def deadline(signum, frame):
        raise TimeoutError("Task smoke reached its 600-second controller budget")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    try:
        backend_port, proxy_port = _free_port(), _free_port()
        servers.append(_serve(_backend(policy, private, rewards), backend_port))
        servers.append(_serve(create_app(route_dir), proxy_port))
        evidence.update(stage="quicktunnel")
        save()
        tunnel_log = private / "tunnel.log"
        with tunnel_log.open("w") as stream:
            tunnel = subprocess.Popen(
                [
                    args.cloudflared,
                    "tunnel",
                    "--url",
                    f"http://127.0.0.1:{proxy_port}",
                    "--no-autoupdate",
                    "--protocol",
                    "http2",
                ],
                stdout=stream,
                stderr=stream,
            )
        until = time.monotonic() + 45
        origin = None
        while time.monotonic() < until:
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", tunnel_log.read_text())
            if match:
                origin = match[0]
                break
            if tunnel.poll() is not None:
                break
            time.sleep(0.25)
        if origin is None:
            raise RuntimeError("Quicktunnel did not become available")
        os.environ.update(
            DSH_GATEWAY_PUBLIC_ORIGIN=origin,
            DSH_GATEWAY_ROUTE_DIR=str(route_dir),
            UNI_AGENT_LOG_DIR=str(private / "agent"),
        )
        session_id = "smoke-" + secrets.token_hex(8)
        config = {
            "model": {"model_name": "policy"},
            "agent": {"type": "dsh-sdk", "run_timeout": 180, "max_tokens": 4096},
            "environment": {
                "environment_class": "modal",
                "git_leak_prevention": "strip",
                "anti_hack_cleanup": True,
                "registry_secret": "mimo-dsh-ghcr-20260928",
                "cpu": 1,
                "memory": 4096,
                "sandbox_timeout": 600,
                "app_name": "train-p0-dsh-task-" + session_id,
                "tags": {"project": "train-p0-integration", "purpose": "dsh-task-smoke"},
            },
        }
        config_path = private / "harness.yaml"
        config_path.write_text(yaml.safe_dump(config))
        spec = private / "spec.yaml"
        spec.write_text(yaml.safe_dump({"harnesses": [{"label": "dsh-sdk", "config": str(config_path)}]}))
        os.environ.update(MIXED_HARNESS_SPEC=str(spec), MIXED_HARNESS_MODE="step-hash")
        session = SimpleNamespace(
            session_id=session_id,
            base_url=f"http://127.0.0.1:{backend_port}/sessions/{session_id}/v1",
            reward_info_url=f"http://127.0.0.1:{backend_port}/sessions/{session_id}/reward",
        )
        evidence.update(stage="sdk-task-grader")
        save()
        asyncio.run(
            mimoagent_runner(
                raw_prompt=SMOKE_PROMPT,
                session=session,
                sample_index=0,
                tools_kwargs={"instance": load_instance(args.row), "harness_round": 0},
                environment_overrides={"environment_class": ObservedModal},
            )
        )
        if not policy.tool_result_verified or policy.calls < 2:
            raise RuntimeError("No real SDK tool round trip was observed")
        final = [item["reward_info"] for item in rewards if "reward" in item.get("reward_info", {})]
        if len(final) != 1 or final[0].get("error_category"):
            raise RuntimeError("Original grader did not return a valid task verdict")
        value = final[0]
        evidence["verdict_classification"] = classify_verdict(value, load_instance(args.row))
        evidence["reward"] = {
            k: value[k]
            for k in (
                "reward",
                "agent_type",
                "agent_status",
                "agent_completed",
                "verifier_returncode",
                "resolved",
                "test_duration",
                "selected_harness",
            )
            if k in value
        }
        traces = list((private / "agent").rglob("dsh-session.jsonl"))
        if len(traces) != 1:
            raise RuntimeError("Expected exactly one private SDK trace")
        trace = traces[0].read_bytes()
        evidence.update(
            status="passed",
            trace_sha256=hashlib.sha256(trace).hexdigest(),
            trace_event_count=len(trace.splitlines()),
            route_revoked=not list(route_dir.glob("*.json")),
        )
    except BaseException as exc:
        evidence.update(status="failed", error_type=type(exc).__name__)
        (private / "exception.txt").write_text(str(exc))
    finally:
        signal.alarm(30)
        cleanup_errors = []
        for instance in instances:
            if instance.sandbox is not None:
                try:
                    instance.cleanup()
                except Exception as exc:
                    cleanup_errors.append(type(exc).__name__)
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait(timeout=5)
        for server, thread in servers:
            server.should_exit = True
            thread.join(timeout=5)
        evidence["tunnel_terminated"] = tunnel is None or tunnel.poll() is not None
        evidence["local_servers_stopped"] = all(not thread.is_alive() for _, thread in servers)
        if cleanup_errors:
            evidence.update(status="failed", cleanup_errors=cleanup_errors)
        signal.alarm(0)
        evidence.update(
            model_requests=policy.calls,
            tool_result_verified=policy.tool_result_verified,
            tool_name=policy.tool_name,
            reward_posts=len(rewards),
            first_request_sizes=policy.first_request_sizes,
            tokenizer_measured=False,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
        if args.tokenizer and (private / "request-1.json").exists():
            try:
                evidence["token_counts"] = measure_prompt_tokens(
                    private / "request-1.json", args.tokenizer, load_instance(args.row)
                )
                evidence["tokenizer_measured"] = True
            except Exception as exc:
                evidence["tokenizer_error_type"] = type(exc).__name__
                (private / "tokenizer-error.txt").write_text(str(exc))
        save()
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cloudflared", default="cloudflared")
    parser.add_argument("--tokenizer", help="Existing local fixed tokenizer directory; CPU only")
    args = parser.parse_args()
    result = run(args)
    raise SystemExit(
        0
        if result["status"] == "passed"
        and result.get("terminated")
        and result.get("tunnel_terminated")
        and result.get("local_servers_stopped")
        else 1
    )


if __name__ == "__main__":
    main()
