# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Recipe-local history stripping fixes the pinned upstream's unused option."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    source = Path(__file__).parents[3] / "third_party/mimoagent-osr/src"
    monkeypatch.syspath_prepend(str(source))
    return importlib.import_module("recipes.code.code_environment")


def instance(kind="opensource-code"):
    return {
        "dataset_type": kind,
        "instance_id": "code-1",
        "cwd": "/testbed",
        "docker_image": "example/image:task",
        "problem_statement": "Fix it",
        "test_patch": "test patch",
        "test_command": "true",
        "verifier_timeout_sec": 10,
    }


def prepared(module, monkeypatch, *, strip_success=True, changed_head=False, changed_tree=False):
    env = module.RuntimeStrippedCodeEnvironment(SimpleNamespace(), instance())
    calls = []
    refs = iter(["a" * 40, ("b" if changed_head else "a") * 40])
    states = iter(["original", "modified" if changed_tree else "original"])
    monkeypatch.setattr(env, "git_add_safe_directory", lambda: calls.append("safe"))
    monkeypatch.setattr(env, "_ensure_work_tree", lambda: calls.append("ensure"))

    def capture():
        calls.append("capture")
        return next(refs)

    def tree_state():
        calls.append("tree")
        return next(states)

    def strip(base):
        calls.append(("strip", base))
        return strip_success

    monkeypatch.setattr(env, "_capture_base_ref", capture)
    monkeypatch.setattr(env, "_capture_worktree_state", tree_state)
    monkeypatch.setattr(env, "_strip_future_commits", strip)
    monkeypatch.setattr(env, "_assert_history_truncated", lambda: calls.append("assert"))
    monkeypatch.setattr(env, "_hide_git", lambda: pytest.fail("No silent hide fallback"))
    return env, calls


def test_setup_strips_captured_base_then_verifies_history_and_tree(module, monkeypatch):
    env, calls = prepared(module, monkeypatch)
    env._setup_dataset_specific()
    assert calls == ["safe", "ensure", "capture", "tree", ("strip", "a" * 40), "assert", "capture", "tree"]
    assert env._base_ref == "a" * 40


def test_strip_failure_stops_setup_without_hiding_git(module, monkeypatch):
    env, calls = prepared(module, monkeypatch, strip_success=False)
    with pytest.raises(RuntimeError, match="strip"):
        env._setup_dataset_specific()
    assert "assert" not in calls


@pytest.mark.parametrize("change", ["changed_head", "changed_tree"])
def test_strip_cannot_change_task_state(module, monkeypatch, change):
    env, _ = prepared(module, monkeypatch, **{change: True})
    with pytest.raises(RuntimeError, match="changed"):
        env._setup_dataset_specific()


def test_factory_uses_copy_and_unique_registry_entry(module, monkeypatch):
    row = instance()
    captured = []
    sentinel = object()
    monkeypatch.setattr(
        module, "make_dataset_env", lambda value, **config: captured.append((value, config)) or sentinel
    )
    original = module.DATASET_REGISTRY["opensource-code"]
    assert module.make_code_dataset_env(row, git_leak_prevention="strip", environment_class="modal") is sentinel
    copied, options = captured[0]
    assert row["dataset_type"] == "opensource-code"
    assert copied is not row
    assert copied["dataset_type"] == "opensource-code-runtime-strip"
    assert copied["original_dataset_type"] == "opensource-code"
    assert options == {"git_leak_prevention": "strip", "environment_class": "modal"}
    assert module.DATASET_REGISTRY["opensource-code"] is original
    assert module.DATASET_REGISTRY[copied["dataset_type"]] is module.RuntimeStrippedCodeEnvironment


@pytest.mark.parametrize("kind,mode", [("opensource-code", "none"), ("opensource-code", "hide"), ("arvo", "strip")])
def test_unaffected_routes_use_original_factory_input(module, monkeypatch, kind, mode):
    row = instance(kind)
    captured = []
    monkeypatch.setattr(module, "make_dataset_env", lambda value, **config: captured.append(value))
    module.make_code_dataset_env(row, git_leak_prevention=mode)
    assert captured == [row]
    assert captured[0] is row


def test_registry_collision_fails_closed(module, monkeypatch):
    monkeypatch.setitem(module.DATASET_REGISTRY, "opensource-code-runtime-strip", object)
    with pytest.raises(RuntimeError, match="registered"):
        module.make_code_dataset_env(instance(), git_leak_prevention="strip")


def test_worktree_snapshot_command_failure_is_infrastructure_failure(module, monkeypatch):
    env = module.RuntimeStrippedCodeEnvironment(SimpleNamespace(), instance())
    monkeypatch.setattr(env, "execute", lambda *args, **kwargs: {"returncode": 1, "output": "git failed"})
    with pytest.raises(RuntimeError, match="worktree"):
        env._capture_worktree_state()


@pytest.fixture
def real_git_environment(module, tmp_path):
    """Execute the actual vendor strip; isolate git config and use GNU date on macOS."""
    import os
    import shutil
    import stat
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    process_env = dict(os.environ, GIT_CONFIG_GLOBAL=str(tmp_path / "gitconfig"), GIT_CONFIG_NOSYSTEM="1")
    date = subprocess.run(["date", "--version"], capture_output=True)
    if date.returncode:
        gdate = shutil.which("gdate")
        if not gdate:
            pytest.skip("Actual vendor strip requires GNU date (install coreutils on macOS)")
        tools = tmp_path / "bin"
        tools.mkdir()
        (tools / "date").symlink_to(gdate)
        process_env["PATH"] = str(tools) + os.pathsep + process_env["PATH"]

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, env=process_env, stderr=subprocess.DEVNULL).decode()

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "core.filemode", "true")
    git("config", "core.abbrev", "10")
    for name in ("deleted.txt", "tracked.txt", "executable.sh"):
        (repo / name).write_text(name + "\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD").strip()
    (repo / "tracked.txt").write_text("future commit\n")
    git("commit", "-qam", "future")
    future = git("rev-parse", "HEAD").strip()
    git("checkout", "--detach", base)
    (repo / "deleted.txt").unlink()
    (repo / "tracked.txt").write_text("intentional dirty task state\n")
    (repo / "executable.sh").chmod(0o755)
    (repo / "untracked.txt").write_text("untracked task state\n")
    (repo / "untracked-link").symlink_to("tracked.txt")

    class LocalShell:
        inject_mutation = False
        strip_executed = False

        def execute(self, command, cwd=None, timeout=30):
            result = subprocess.run(
                ["bash", "-c", command], cwd=cwd or repo, env=process_env, capture_output=True, timeout=timeout
            )
            if "gc --prune=now" in command:
                self.strip_executed = True
                # Deterministic equivalent of auto-abbreviation shortening after object pruning.
                git("config", "core.abbrev", "9")
                if self.inject_mutation:
                    (repo / "tracked.txt").write_text("unexpected task mutation\n")
            return {"returncode": result.returncode, "output": (result.stdout + result.stderr).decode()}

    def manifest():
        values = {}
        for path in repo.iterdir():
            if path.name == ".git":
                continue
            values[path.name] = (
                stat.S_IMODE(path.lstat().st_mode),
                ("link", os.readlink(path)) if path.is_symlink() else ("file", path.read_bytes()),
            )
        return values

    shell = LocalShell()
    env = module.RuntimeStrippedCodeEnvironment(shell, {**instance(), "cwd": str(repo)})
    return env, shell, git, manifest, base, future


def test_real_vendor_strip_keeps_dirty_tree_despite_shorter_oid_display(real_git_environment):
    env, shell, git, manifest, base, future = real_git_environment
    before = manifest()
    original_display = git("diff", "--no-ext-diff", "--binary", "HEAD", "--")
    stable_display = env._capture_worktree_state()
    env._setup_dataset_specific()
    assert shell.strip_executed
    assert git("config", "core.abbrev").strip() == "9"
    assert git("diff", "--no-ext-diff", "--binary", "HEAD", "--") != original_display
    assert env._capture_worktree_state() == stable_display
    assert manifest() == before
    assert git("rev-parse", "HEAD").strip() == base
    assert git("rev-list", "--all", "--not", base) == ""
    assert shell.execute(f"git cat-file -e {future}")["returncode"] != 0
    assert " D deleted.txt" in stable_display
    assert "?? untracked.txt" in stable_display
    assert "?? untracked-link" in stable_display
    assert "old mode 100644" in stable_display and "new mode 100755" in stable_display


def test_real_vendor_strip_still_rejects_actual_file_mutation(real_git_environment):
    env, shell, _, _, _, _ = real_git_environment
    shell.inject_mutation = True
    with pytest.raises(RuntimeError, match="changed task worktree"):
        env._setup_dataset_specific()
    assert shell.strip_executed
