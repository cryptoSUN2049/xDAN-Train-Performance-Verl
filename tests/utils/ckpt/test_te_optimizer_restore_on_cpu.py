"""Checkpoint staging preserves dtype, aliases and non-parameter state."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[3] / "verl/utils/checkpoint/te_optimizer_restore.py"
spec = importlib.util.spec_from_file_location("te_optimizer_restore", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pre_hook_changes_shared_nested_state_but_preserves_original_mapping():
    state = {"exp_avg": torch.ones(3), "exp_avg_sq": torch.ones(3), "master_param": torch.ones(3, dtype=torch.int16)}
    original = {"state": {0: state, "metadata": {"step": 7}}, "param_groups": [{"params": [0], "step": 2}]}
    torch_input = original.copy()
    result = module._stage_te_state(None, torch_input)
    assert result["state"] == {"metadata": {"step": 7}}
    assert original["state"][0] is state
    assert original["state"][0]["master_param"].dtype == torch.int16
    assert original["state"][0]["exp_avg"].dtype == torch.float32
    assert result["param_groups"][0]["step"] == 2


@pytest.mark.parametrize("bad", [torch.ones(2, dtype=torch.bfloat16), torch.nn.Parameter(torch.ones(2))])
def test_reject_unsupported_state_before_mutation(bad):
    original = {"state": {0: {"exp_avg": bad}}, "param_groups": [{"params": [0]}]}
    with pytest.raises(TypeError):
        module._stage_te_state(None, original.copy())
    assert original["state"][0]["exp_avg"] is bad


def test_stage_preserves_alias_and_rejects_tensor_subclass():
    tensor = torch.ones(2)
    result = module.optimizer_state_to_cpu({"a": tensor, "b": [tensor]})
    assert result["a"] is result["b"][0]
    with pytest.raises(TypeError):
        module.optimizer_state_to_cpu(torch.nn.Parameter(tensor))


def test_chained_leaf_discovery():
    leaves = [object(), object()]
    wrapped = SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=leaf) for leaf in leaves])
    assert module._leaves(wrapped) == leaves


def test_scoped_hook_removes_on_failure_and_rejects_conflicting_hooks(monkeypatch):
    import sys

    class FakeFusedAdam(torch.optim.Optimizer):
        def __init__(self):
            super().__init__([torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))], {})
            self.master_weights = True
            self.store_param_remainders = True
            self.capturable = False
            self.name_to_dtype_map = {name: torch.float32 for name in ("exp_avg", "exp_avg_sq", "master_param")}

    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.optimizers", SimpleNamespace(FusedAdam=FakeFusedAdam))
    monkeypatch.setattr(module, "version", lambda package: "2.16.1+audited")
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")
    optimizer = FakeFusedAdam()
    with pytest.raises(ValueError, match="restore failed"):
        with module.stage_te_optimizer_restore(optimizer) as enabled:
            assert enabled and len(optimizer._optimizer_load_state_dict_pre_hooks) == 1
            raise ValueError("restore failed")
    assert not optimizer._optimizer_load_state_dict_pre_hooks
    handle = optimizer.register_load_state_dict_pre_hook(lambda *args: None)
    with pytest.raises(RuntimeError, match="existing load-state hooks"):
        with module.stage_te_optimizer_restore(optimizer):
            pytest.fail("conflicting hook should prevent staging")
    assert len(optimizer._optimizer_load_state_dict_pre_hooks) == 1
    handle.remove()
