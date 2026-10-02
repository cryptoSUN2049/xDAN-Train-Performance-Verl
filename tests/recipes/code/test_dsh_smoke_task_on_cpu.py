"""The live smoke must prove returned tool output, never manufacture a reward."""

import json

import pytest

from scripts.code.dsh.smoke_task import ScriptedPolicy, load_instance


def test_scripted_policy_demands_a_real_round_trip():
    policy = ScriptedPolicy("nonce-123")
    request = {
        "tools": [
            {
                "type": "function",
                "function": {"name": "bash", "parameters": {"properties": {"command": {"type": "string"}}}},
            }
        ],
        "messages": [],
    }
    message, finish = policy.respond(request)
    assert finish == "tool_calls"
    assert "cat /etc/os-release" in message["tool_calls"][0]["function"]["arguments"]
    with pytest.raises(ValueError, match="tool result"):
        policy.respond({"messages": [{"role": "assistant", "content": "nonce-123"}]})


def test_scripted_policy_accepts_nonce_only_from_tool_response():
    policy = ScriptedPolicy("nonce-123")
    policy.calls = 1
    message, finish = policy.respond({"messages": [{"role": "tool", "content": "nonce-123\nref: refs/heads/main"}]})
    assert finish == "stop"
    assert policy.tool_result_verified
    assert message["role"] == "assistant"


def test_echoed_command_error_does_not_prove_tool_execution():
    policy = ScriptedPolicy("nonce-123")
    policy.calls = 1
    with pytest.raises(ValueError, match="tool result"):
        policy.respond({"messages": [{"role": "tool", "content": "Could not run: printf nonce-123"}]})


def test_source_row_retains_original_grader(tmp_path):
    instance = {
        "instance_id": "format-code-task-001661",
        "test_patch": "hidden",
        "test_command": "bash /workspace/repo/mimo_test_command.sh",
        "verifier_timeout_sec": 1800,
    }
    source = tmp_path / "row.json"
    source.write_text(json.dumps({"extra_info": {"instance_json": json.dumps(instance)}}))
    parsed = load_instance(source)
    assert parsed["test_patch"] == instance["test_patch"]
    assert parsed["test_command"] == instance["test_command"]
    assert parsed["verifier_timeout_sec"] == 180
    assert "@sha256:" in parsed["docker_image"]


def test_backend_stream_and_reward_handshake(tmp_path):
    import asyncio

    import httpx

    from scripts.code.dsh.smoke_task import _backend

    policy = ScriptedPolicy("nonce")
    rewards = []
    app = _backend(policy, tmp_path, rewards)

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/sessions/s1/v1/chat/completions",
                json={
                    "stream": True,
                    "tools": [{"function": {"name": "bash", "parameters": {"properties": {"command": {}}}}}],
                },
            )
            assert response.status_code == 200
            assert response.text.endswith("data: [DONE]\n\n")
            assert '"finish_reason": "tool_calls"' in response.text
            response = await client.post("/sessions/s1/reward", json={"reward_info": {"reward": 0}})
            assert response.status_code == 200
            assert rewards == [{"reward_info": {"reward": 0}}]

    asyncio.run(check())


def test_expected_missing_task_api_is_not_missing_dependency():
    from scripts.code.dsh.smoke_task import classify_verdict

    instance = {"instance_id": "format-code-task-001661", "problem_statement": "Please add friend_set_for(user)"}
    assert (
        classify_verdict(
            {"verifier_returncode": 2, "test_output": "cannot import name 'friend_set_for' from 'friends.models'"},
            instance,
        )
        == "required_task_api_not_implemented"
    )
    with pytest.raises(RuntimeError):
        classify_verdict({"verifier_returncode": 2, "test_output": "No module named django"}, instance)
    with pytest.raises(RuntimeError):
        classify_verdict({"verifier_returncode": 124}, instance)


def test_token_counter_counts_ids_not_batchencoding_keys(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from scripts.code.dsh.smoke_task import SMOKE_PROMPT, measure_prompt_tokens

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["tools"] == [{"name": "bash"}]
            return {
                "input_ids": list(range(42 if messages[0]["content"] == SMOKE_PROMPT else 87)),
                "attention_mask": [],
            }

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())),
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"messages": [{"role": "user", "content": SMOKE_PROMPT}], "tools": [{"name": "bash"}]})
    )
    result = measure_prompt_tokens(request, "/fixed", {"problem_statement": "Original task"})
    assert result["scripted_prompt_tokens"] == 42
    assert result["original_task_prompt_tokens"] == 87
    assert result["original_task_total_with_completion"] == 4183
