# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""CPU contract tests for the Harbor task environment and its data preparation."""

from __future__ import annotations

import base64
import io
import json
import os
import tarfile

import pytest

from recipes.harbor import environment as harbor
from scripts.harbor import prepare_data


class FakeSandbox:
    """Minimal Environment double: a dict filesystem plus a scripted test.sh."""

    def __init__(self, *, verdict: dict | None = None, test_reason: str = "ok"):
        self.files: dict[str, str] = {}
        self.commands: list[tuple[str, str | None]] = []
        self.verdict = verdict if verdict is not None else {"reward.txt": "1\n"}
        self.test_reason = test_reason
        self.config = type("Config", (), {"answer_leak_blocklist": None})()

    def start(self):
        pass

    def execute(self, command, cwd=None, timeout=None, **_):
        self.commands.append((command, cwd))
        if command.startswith("rm -rf"):
            for path in list(self.files):
                if path.startswith(("/tests", "/logs/verifier")):
                    del self.files[path]
            return {"returncode": 0, "output": "", "reason": "ok"}
        if command.startswith("bash /tests/test.sh"):
            if self.test_reason != "ok":
                return {"returncode": 124, "output": "", "reason": self.test_reason}
            assert "/tests/test.sh" in self.files, "tests were not uploaded before running"
            self.files["/logs/verifier/test-stdout.txt"] = "1 passed\n"
            for name, content in self.verdict.items():
                self.files[f"/logs/verifier/{name}"] = content
            return {"returncode": 0, "output": "", "reason": "ok"}
        if command.startswith("test -f "):
            path = command.split()[2]
            if path in self.files:
                return {"returncode": 0, "output": self.files[path], "reason": "ok"}
            return {"returncode": 1, "output": "", "reason": "ok"}
        return {"returncode": 0, "output": "", "reason": "ok"}

    def copy_to(self, src, dest):
        for root, _dirs, names in os.walk(src):
            for name in names:
                rel = os.path.relpath(os.path.join(root, name), src)
                with open(os.path.join(root, name)) as handle:
                    self.files[f"{dest}/{rel}"] = handle.read()


def _tests_b64() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        data = b"#!/bin/bash\necho run\n"
        info = tarfile.TarInfo("test.sh")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return base64.b64encode(buffer.getvalue()).decode()


def _env(sandbox: FakeSandbox, **instance_overrides) -> harbor.HarborEnvironment:
    instance = {
        "dataset_type": "harbor",
        "instance_id": "fix-git",
        "docker_image": "alexgshaw/fix-git:20251031",
        "cwd": "/app/personal-site",
        "verifier_timeout_sec": 30.0,
        "tests_tar_b64": _tests_b64(),
        **instance_overrides,
    }
    return harbor.HarborEnvironment(sandbox, instance)


def test_reward_txt_verdict_and_working_directory():
    sandbox = FakeSandbox()
    env = _env(sandbox)
    env.setup_environment()
    reward, _output, extra = env.calculate_reward()
    assert reward == 1.0
    assert extra["harbor_reward"] == 1.0 and "error_category" not in extra
    run_cwd = [cwd for command, cwd in sandbox.commands if command.startswith("bash /tests/test.sh")]
    assert run_cwd == ["/app/personal-site"]


def test_reward_json_wins_over_reward_txt():
    sandbox = FakeSandbox(verdict={"reward.json": json.dumps({"reward": 0.25}), "reward.txt": "1"})
    reward, _output, extra = _env(sandbox).calculate_reward()
    assert reward == 0.25 and extra["harbor_reward"] == 0.25


def test_missing_reward_is_infra_not_zero_training_signal():
    sandbox = FakeSandbox(verdict={})
    _reward, _output, extra = _env(sandbox).calculate_reward()
    assert extra["error_category"] == harbor.REWARD_MISSING
    assert "harbor_reward" not in extra


def test_verifier_timeout_is_ungradable():
    sandbox = FakeSandbox(test_reason="pod_timeout")
    _reward, _output, extra = _env(sandbox).calculate_reward()
    assert extra["error_category"] == harbor.VERIFIER_EXEC_FAILED


def test_preplanted_verdict_is_wiped_before_grading():
    sandbox = FakeSandbox(verdict={})
    sandbox.files["/logs/verifier/reward.txt"] = "1"  # agent tried to plant a pass
    _reward, _output, extra = _env(sandbox).calculate_reward()
    assert extra["error_category"] == harbor.REWARD_MISSING


def test_grading_never_stages_the_agent_repository():
    sandbox = FakeSandbox()
    _env(sandbox).calculate_reward()
    assert not any("git add" in command for command, _cwd in sandbox.commands)


def test_missing_tests_are_classified():
    sandbox = FakeSandbox()
    env = _env(sandbox, tests_tar_b64=None, task_path="/nonexistent")
    _reward, _output, extra = env.calculate_reward()
    assert extra["error_category"] == harbor.TESTS_UNAVAILABLE


def test_runner_rejects_ungradable_harbor_rollouts():
    from recipes.code.mimoagent_runner import _validate_code_reward

    instance = {"dataset_type": "harbor"}
    _validate_code_reward(instance, {"harbor_reward": 0.0})  # a real failing verdict trains as 0
    with pytest.raises(RuntimeError):
        _validate_code_reward(instance, {"error_category": harbor.REWARD_MISSING})
    with pytest.raises(RuntimeError):
        _validate_code_reward(instance, {"transport_error": True})
    with pytest.raises(RuntimeError):
        _validate_code_reward(instance, {})


def test_register_is_idempotent():
    harbor.register()
    harbor.register()
    from mimoagent.environments.datasets import DATASET_REGISTRY

    assert DATASET_REGISTRY["harbor"] is harbor.HarborEnvironment


def _make_task(root, name, *, workdir="/app", image="alexgshaw/demo:1"):
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/bash\necho ok\n")
    (task / "instruction.md").write_text(f"Do {name}.\n")
    (task / "environment" / "Dockerfile").write_text(f"FROM ubuntu\nWORKDIR /tmp\nWORKDIR {workdir}\n")
    image_line = f'docker_image = "{image}"\n' if image else ""
    (task / "task.toml").write_text(
        f"[verifier]\ntimeout_sec = 120.0\n[agent]\ntimeout_sec = 600.0\n[environment]\n{image_line}cpus = 1\n"
    )
    return task


def test_prepare_data_row_matches_code_recipe_contract(tmp_path):
    _make_task(tmp_path, "fix-git", workdir="/app/personal-site")
    row = prepare_data.task_row(tmp_path / "fix-git", 0)
    instance = json.loads(row["extra_info"]["instance_json"])
    assert row["prompt"] == [{"role": "user", "content": "Do fix-git.\n"}]
    assert instance["problem_statement"] == "Do fix-git.\n"
    assert instance["dataset_type"] == "harbor" and row["extra_info"]["dataset_type"] == "harbor"
    assert instance["cwd"] == "/app/personal-site"  # last WORKDIR wins
    assert instance["docker_image"] == "alexgshaw/demo:1"
    assert instance["verifier_timeout_sec"] == 120.0
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(instance["tests_tar_b64"])), mode="r:gz") as archive:
        assert "test.sh" in archive.getnames()


def test_prepare_data_refuses_tasks_without_prebuilt_image(tmp_path):
    _make_task(tmp_path, "needs-build", image=None)
    with pytest.raises(ValueError, match="no prebuilt docker_image"):
        prepare_data.task_row(tmp_path / "needs-build", 0)


def test_built_image_map_fills_tasks_without_prebuilt_image(tmp_path):
    from scripts.harbor.build_images import context_sha256

    task = _make_task(tmp_path, "stage1-task", image=None)
    image_map = {"stage1-task": {"image": "im-abc123", "env_sha256": context_sha256(task / "environment")}}
    instance = json.loads(prepare_data.task_row(task, 0, image_map)["extra_info"]["instance_json"])
    assert instance["docker_image"] == "im-abc123"


def test_stale_built_image_is_rejected(tmp_path):
    task = _make_task(tmp_path, "stage1-task", image=None)
    image_map = {"stage1-task": {"image": "im-abc123", "env_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="stale"):
        prepare_data.task_row(task, 0, image_map)


def test_declared_runtime_workdir_wins_over_dockerfile(tmp_path):
    task = _make_task(tmp_path, "nox", workdir="/app")
    (task / "task.toml").write_text((task / "task.toml").read_text() + '[harbor_runtime]\nworkdir = "/nox"\n')
    instance = json.loads(prepare_data.task_row(task, 0)["extra_info"]["instance_json"])
    assert instance["cwd"] == "/nox"


def test_prepared_row_round_trips_through_harbor_environment(tmp_path):
    _make_task(tmp_path, "regex-log")
    instance = json.loads(prepare_data.task_row(tmp_path / "regex-log", 0)["extra_info"]["instance_json"])
    sandbox = FakeSandbox(verdict={"reward.txt": "0"})
    reward, _output, extra = harbor.HarborEnvironment(sandbox, instance).calculate_reward()
    assert reward == 0.0 and extra["harbor_reward"] == 0.0
    assert sandbox.files["/tests/test.sh"].startswith("#!/bin/bash")
