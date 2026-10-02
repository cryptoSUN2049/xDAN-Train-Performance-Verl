# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Harbor task environment for MiMo-Agent.

A Harbor task is ``task.toml`` + ``instruction.md`` + ``tests/`` + a container image. Any harness
(MiMo-Agent native loops, DSH, blackbox CLIs) works inside the task container; grading reproduces
Harbor's own verifier contract after the agent finishes:

1. wipe ``/tests`` and ``/logs/verifier`` so the agent cannot pre-plant a verdict,
2. upload the task's ``tests/`` to ``/tests``,
3. run ``bash /tests/test.sh`` in the task's working directory,
4. read ``/logs/verifier/reward.json`` (preferred) or ``reward.txt``.

A verifier that cannot produce a verdict is an infrastructure failure, not a negative example: it
is reported through ``error_category`` so the runner drops the rollout instead of training on 0.
"""

from __future__ import annotations

import base64
import io
import json
import os
import tarfile
import tempfile

from mimoagent.environments.datasets import DATASET_REGISTRY
from mimoagent.environments.datasets.base import DatasetEnvironment

DATASET_TYPE = "harbor"
VERIFIER_LOG_DIR = "/logs/verifier"
TESTS_DIR = "/tests"
DEFAULT_VERIFIER_TIMEOUT_SEC = 900.0

# error_category values; all of them mean "ungradable", see recipes/code/mimoagent_runner.py.
REWARD_MISSING = "harbor_reward_missing"
REWARD_UNPARSABLE = "harbor_reward_unparsable"
VERIFIER_EXEC_FAILED = "harbor_verifier_exec_failed"
TESTS_UNAVAILABLE = "harbor_tests_unavailable"


def parse_reward(reward_json: str | None, reward_txt: str | None) -> float:
    """Harbor's precedence: ``reward.json`` {"reward": x} first, then the scalar in ``reward.txt``."""
    if reward_json is not None and reward_json.strip():
        payload = json.loads(reward_json)
        if isinstance(payload, dict) and "reward" in payload:
            return float(payload["reward"])
        if isinstance(payload, int | float):
            return float(payload)
        raise ValueError("reward.json has no 'reward' field")
    if reward_txt is not None and reward_txt.strip():
        return float(reward_txt.strip().splitlines()[0])
    raise FileNotFoundError("no reward.json or reward.txt")


class HarborEnvironment(DatasetEnvironment):
    """Run a Harbor task in its own image and grade it with the task's own tests."""

    # The agent's repository state is the deliverable; never rewrite its git history or index.
    _GIT_LEAK_PREVENTION_DEFAULT = "none"
    _ANTI_HACK_CLEANUP_DEFAULT = False

    @classmethod
    def default_docker_image(cls, instance: dict) -> str | None:
        return instance.get("docker_image")

    @property
    def repo_path(self) -> str:
        return self.instance.get("cwd") or "/app"

    def _setup_dataset_specific(self) -> None:
        result = self.execute(f"rm -rf {TESTS_DIR} && mkdir -p {VERIFIER_LOG_DIR} /logs/agent", cwd="/", timeout=120)
        if result.get("returncode") != 0:
            raise RuntimeError(f"{self.instance_id}: harbor setup failed: {str(result.get('output'))[-500:]}")

    def _capture_model_diff(self) -> tuple[str, str]:
        # The base implementation runs `git add -A` in repo_path, which would mutate the graded state
        # of git-centric tasks (e.g. fix-git). Harbor grades the container, not a diff.
        return "", ""

    def _materialize_tests(self, workdir: str) -> str:
        """Return a local directory holding the task's tests/ (inline tarball or shared-volume path)."""
        inline = self.instance.get("tests_tar_b64")
        if inline:
            target = os.path.join(workdir, "tests")
            os.makedirs(target)
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(inline)), mode="r:gz") as archive:
                archive.extractall(target, filter="data")
            return target
        task_path = self.instance.get("task_path")
        if task_path and os.path.isdir(os.path.join(task_path, "tests")):
            return os.path.join(task_path, "tests")
        raise FileNotFoundError("task carries neither tests_tar_b64 nor a readable task_path/tests")

    def _read_remote(self, path: str) -> str | None:
        result = self.execute(f"test -f {path} && cat {path}", cwd="/", timeout=60)
        if result.get("returncode") != 0:
            return None
        return str(result.get("output") or "")

    def _do_calculate_reward(
        self, timeout: int | float | None = None, model_patch: str = ""
    ) -> tuple[float, str, dict]:
        verifier_timeout = float(self.instance.get("verifier_timeout_sec") or DEFAULT_VERIFIER_TIMEOUT_SEC)
        extra: dict = {"harbor_task": self.instance_id}

        wipe = self.execute(
            f"rm -rf {TESTS_DIR} {VERIFIER_LOG_DIR} && mkdir -p {VERIFIER_LOG_DIR}", cwd="/", timeout=120
        )
        if wipe.get("returncode") != 0:
            extra["error_category"] = VERIFIER_EXEC_FAILED
            return 0.0, str(wipe.get("output") or ""), extra

        with tempfile.TemporaryDirectory(prefix="harbor-tests-") as workdir:
            try:
                local_tests = self._materialize_tests(workdir)
            except Exception as error:  # noqa: BLE001 - classified, never trained on
                extra["error_category"] = TESTS_UNAVAILABLE
                return 0.0, repr(error), extra
            self.env.copy_to(local_tests, TESTS_DIR)

        command = f"bash {TESTS_DIR}/test.sh > {VERIFIER_LOG_DIR}/test-stdout.txt 2>&1"
        run = self.execute(command, cwd=self.repo_path, timeout=int(verifier_timeout))
        rc = run.get("returncode")
        reason = run.get("reason")
        extra["verifier_returncode"] = rc
        extra["verifier_reason"] = reason
        stdout = self._read_remote(f"{VERIFIER_LOG_DIR}/test-stdout.txt") or str(run.get("output") or "")
        tail = stdout[-4000:]

        if reason not in (None, "", "ok"):
            # Timeouts / transport loss: the verdict is unknown, so the rollout is ungradable.
            extra["error_category"] = VERIFIER_EXEC_FAILED
            return 0.0, tail, extra

        try:
            reward = parse_reward(
                self._read_remote(f"{VERIFIER_LOG_DIR}/reward.json"),
                self._read_remote(f"{VERIFIER_LOG_DIR}/reward.txt"),
            )
        except FileNotFoundError:
            extra["error_category"] = REWARD_MISSING
            return 0.0, tail, extra
        except (ValueError, json.JSONDecodeError) as error:
            extra["error_category"] = REWARD_UNPARSABLE
            return 0.0, f"{error!r}\n{tail}", extra
        extra["harbor_reward"] = reward
        return reward, tail, extra


def register() -> None:
    """Idempotently register ``dataset_type: harbor`` without touching pinned MiMo-Agent."""
    registered = DATASET_REGISTRY.setdefault(DATASET_TYPE, HarborEnvironment)
    if registered is not HarborEnvironment:
        raise RuntimeError(f"Dataset type {DATASET_TYPE!r} is already registered to {registered!r}")
