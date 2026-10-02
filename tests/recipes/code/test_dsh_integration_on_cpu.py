import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("status", ["Completed", "LimitsExceeded"])
def test_dsh_uses_existing_environment_reward_and_revokes_route(monkeypatch, tmp_path, fail, status):
    from recipes.code import dsh_agent, mimoagent_runner

    events = []
    environment = SimpleNamespace(
        env=object(),
        setup_environment=lambda: events.append("setup"),
        calculate_reward=lambda: (events.append("grade") or 1.0, "passed", {}),
        cleanup=lambda: events.append("cleanup"),
    )
    factory = ModuleType("mimoagent.agents.factory")
    factory.get_agent_class = lambda name: pytest.fail(f"DSH SDK must not replace upstream factory: {name}")
    environments = ModuleType("mimoagent.environments.utils")
    environments.make_dataset_env = lambda *args, **kwargs: environment
    monkeypatch.setitem(sys.modules, "mimoagent.agents.factory", factory)
    monkeypatch.setitem(sys.modules, "mimoagent.environments.utils", environments)
    route_dir = tmp_path / "routes"
    monkeypatch.setenv("DSH_GATEWAY_PUBLIC_ORIGIN", "https://our-gateway.example")
    monkeypatch.setenv("DSH_GATEWAY_ROUTE_DIR", str(route_dir))

    class Agent:
        IDLE_STATUS = "Completed"

        def __init__(self, model, env, **kwargs):
            assert env is environment.env
            self.model = model

        def run(self, task):
            assert task == "fix it"
            assert self.model.config.model_kwargs["base_url"] == "https://our-gateway.example/sessions/test-1/v1"
            assert self.model.config.model_kwargs["api_key"] != "not-needed"
            assert len(list(route_dir.glob("*.json"))) == 1
            events.append("agent")
            if fail:
                raise RuntimeError("infra")
            return status, "fixed"

    monkeypatch.setattr(dsh_agent, "DshSdkAgent", Agent)
    monkeypatch.setattr(mimoagent_runner, "_notify_agent_finished", lambda *args, **kwargs: events.append("notify"))
    kwargs = dict(
        raw_prompt="fix it",
        instance={},
        session=SimpleNamespace(session_id="test-1", base_url="http://127.0.0.1:1234/sessions/test-1/v1"),
        config={"agent": {"type": "dsh-sdk"}, "environment": {"environment_class": "modal"}},
        agent_overrides={},
        environment_overrides={},
    )
    if fail:
        with pytest.raises(RuntimeError, match="infra"):
            mimoagent_runner._run_sync(**kwargs)
        assert events == ["setup", "agent", "cleanup"]
    else:
        reward = mimoagent_runner._run_sync(**kwargs)
        assert reward["agent_type"] == "dsh-sdk"
        assert reward["reward"] == 1.0
        assert reward["agent_completed"] is (status == "Completed")
        assert reward["termination_kind"] == ("completed" if status == "Completed" else "truncated")
        assert events == ["setup", "agent", "notify", "grade", "cleanup"]
    assert list(route_dir.glob("*.json")) == []


@pytest.mark.parametrize(
    "extra",
    [
        {"error_category": "reward/testbed_corrupted"},
        {"transport_error": True},
        {},
        {"verifier_returncode": None},
        {"verifier_returncode": -1},
        {"verifier_returncode": 124},
    ],
)
def test_infra_grading_cannot_become_a_zero_reward_sample(extra):
    from recipes.code.mimoagent_runner import _validate_code_reward

    with pytest.raises(RuntimeError, match="verifier"):
        _validate_code_reward({"dataset_type": "opensource-code"}, extra)


@pytest.mark.parametrize("rc", [0, 1, 2])
def test_real_test_exit_codes_remain_gradable(rc):
    from recipes.code.mimoagent_runner import _validate_code_reward

    _validate_code_reward({"dataset_type": "opensource-code"}, {"verifier_returncode": rc})
