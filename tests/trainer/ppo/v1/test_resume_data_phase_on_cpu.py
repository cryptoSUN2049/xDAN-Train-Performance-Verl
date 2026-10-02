"""Checkpoint model restoration remains mandatory when switching sync datasets."""

import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "verl/trainer/ppo/v1/trainer_base.py"


class Config(dict):
    __getattr__ = dict.__getitem__


def restore(tmp_path, *, mode="sync", override=None):
    path = tmp_path / "global_step_2"
    path.mkdir()
    (path / "data.pt").write_bytes(b"fixture-not-unpickled")
    calls = []
    config = Config(resume_mode="resume_path", resume_from_path=str(path), del_local_ckpt_after_load=False)
    if override is not None:
        config["resume_dataloader"] = override
    trainer = SimpleNamespace(
        config=SimpleNamespace(trainer=config),
        trainer_mode=mode,
        use_critic=False,
        actor_rollout_wg=SimpleNamespace(load_checkpoint=lambda **kwargs: calls.append(("actor", kwargs))),
        train_dataloader=SimpleNamespace(load_state_dict=lambda state: calls.append(("data", state))),
    )
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PPOTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_load_checkpoint")
    namespace = {
        "os": os,
        "torch": SimpleNamespace(load=lambda *args, **kwargs: {"position": 2}),
        "logger": logging.getLogger(__name__),
        "_tq_supports_checkpoint": lambda: False,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    namespace["_load_checkpoint"](trainer)
    return trainer, calls


def test_default_restores_data_and_actor(tmp_path):
    trainer, calls = restore(tmp_path)
    assert trainer.global_steps == 2
    assert [kind for kind, value in calls] == ["actor", "data"]
    assert calls[0][1]["local_path"] == str(tmp_path / "global_step_2/actor")
    assert calls[0][1]["del_local_after_load"] is False


def test_explicit_new_data_still_restores_actor_and_step(tmp_path):
    trainer, calls = restore(tmp_path, override=False)
    assert trainer.global_steps == 2
    assert [kind for kind, value in calls] == ["actor"]


@pytest.mark.parametrize("mode", ["colocate_async", "fully_async"])
def test_async_reset_rejected(tmp_path, mode):
    with pytest.raises(ValueError, match="requires sync"):
        restore(tmp_path, mode=mode, override=False)


def test_string_false_not_silently_treated_as_true(tmp_path):
    with pytest.raises(ValueError, match="must be a boolean"):
        restore(tmp_path, override="False")
