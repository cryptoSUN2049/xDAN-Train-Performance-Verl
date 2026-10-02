# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Recipe-local opt-in repair for history stripping in pinned MiMo-Agent.

The upstream Code setup advertises ``git_leak_prevention: strip`` but never
calls its strip helper. Keep upstream immutable and register a distinct runtime
wrapper only for Code rows explicitly requesting that behavior.
"""

from mimoagent.environments.datasets import DATASET_REGISTRY
from mimoagent.environments.datasets.opensource_code import OpenSourceCodeEnvironment
from mimoagent.environments.utils import make_dataset_env

_RUNTIME_DATASET_TYPE = "opensource-code-runtime-strip"


class RuntimeStrippedCodeEnvironment(OpenSourceCodeEnvironment):
    """Strip future history before exposing the task, preserving its base state."""

    def _capture_worktree_state(self):
        result = self.execute(
            "git status --porcelain=v1 --untracked-files=all && git diff --no-ext-diff --binary --full-index HEAD --",
            cwd=self.repo_path,
        )
        if result.get("returncode", 1) != 0:
            raise RuntimeError(f"{self.instance_id}: could not verify task worktree state")
        return str(result.get("output") or "")

    def _setup_dataset_specific(self):
        self.git_add_safe_directory()
        self._ensure_work_tree()
        self._base_ref = self._capture_base_ref()
        original_worktree = self._capture_worktree_state()
        if not self._strip_future_commits(self._base_ref):
            # The upstream dispatcher silently falls back to hiding .git. Code
            # grading needs the captured base, so an unverified strip is fatal.
            raise RuntimeError(f"{self.instance_id}: runtime git history strip failed")
        self._assert_history_truncated()
        if self._capture_base_ref() != self._base_ref:
            raise RuntimeError(f"{self.instance_id}: runtime git history strip changed HEAD")
        if self._capture_worktree_state() != original_worktree:
            raise RuntimeError(f"{self.instance_id}: runtime git history strip changed task worktree")


def make_code_dataset_env(instance, **config):
    """Delegate normal routes unchanged; copy explicitly opted-in Code rows."""
    if instance.get("dataset_type") != "opensource-code" or config.get("git_leak_prevention") != "strip":
        return make_dataset_env(instance, **config)
    registered = DATASET_REGISTRY.setdefault(_RUNTIME_DATASET_TYPE, RuntimeStrippedCodeEnvironment)
    if registered is not RuntimeStrippedCodeEnvironment:
        raise RuntimeError(f"Dataset type {_RUNTIME_DATASET_TYPE!r} is already registered to another class")
    runtime_instance = {
        **instance,
        "original_dataset_type": "opensource-code",
        "dataset_type": _RUNTIME_DATASET_TYPE,
    }
    return make_dataset_env(runtime_instance, **config)
