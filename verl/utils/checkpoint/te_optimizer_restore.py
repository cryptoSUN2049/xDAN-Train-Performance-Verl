"""Bound transient GPU copies in the audited TE 2.16.1 checkpoint loader.

The public Torch pre-hook only stages the input to TE's own restore logic.
No optimizer algorithm, state precision, or installed library is replaced.
"""

from contextlib import contextmanager
from importlib.metadata import version

import torch
from packaging.version import Version


def optimizer_state_to_cpu(value, memo=None):
    """Copy plain tensor leaves to CPU, preserving repeated tensor aliases."""
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        if type(value) is not torch.Tensor:
            raise TypeError("TE checkpoint staging supports ordinary tensors only")
        if id(value) not in memo:
            memo[id(value)] = value.to(device="cpu", non_blocking=False)
        return memo[id(value)]
    if isinstance(value, dict):
        return {key: optimizer_state_to_cpu(item, memo) for key, item in value.items()}
    if isinstance(value, list):
        return [optimizer_state_to_cpu(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(optimizer_state_to_cpu(item, memo) for item in value)
    return value


def _stage_te_state(optimizer, state_dict):
    """Torch shallow-copies this dict; TE still sees its nested state changes."""
    mapped = {key for group in state_dict["param_groups"] for key in group["params"]}
    # Validate the entire input before modifying any caller-owned state.
    for key, fields in state_dict["state"].items():
        if key not in mapped:
            continue
        for name, tensor in fields.items():
            expected = torch.int16 if name == "master_param" else torch.float32
            if name not in {"master_param", "exp_avg", "exp_avg_sq"}:
                raise ValueError(f"Unsupported TE checkpoint state: {name}")
            if type(tensor) is not torch.Tensor or tensor.dtype != expected:
                raise TypeError(f"Unsupported TE checkpoint tensor for {name}")
    for key, fields in state_dict["state"].items():
        if key in mapped:
            for name, tensor in fields.items():
                fields[name] = tensor.to(device="cpu", non_blocking=False)
    # Keep non-parameter state: TE's post-super loop does not restore those keys.
    return {**state_dict, "state": {key: fields for key, fields in state_dict["state"].items() if key not in mapped}}


def _leaves(optimizer):
    if hasattr(optimizer, "chained_optimizers"):
        return [leaf for child in optimizer.chained_optimizers for leaf in _leaves(child)]
    return [getattr(optimizer, "optimizer", optimizer)]


@contextmanager
def stage_te_optimizer_restore(optimizer):
    """Yield whether audited ordinary BF16/FP32/int16 TE staging is active."""
    leaves = _leaves(optimizer)
    try:
        from transformer_engine.pytorch.optimizers import FusedAdam
    except ImportError:
        yield False
        return
    supported = (
        Version(version("transformer-engine")).base_version == "2.16.1"
        and Version(torch.__version__).base_version == "2.11.0"
        and bool(leaves)
    )
    for leaf in leaves:
        supported = supported and (
            type(leaf) is FusedAdam
            and leaf.master_weights
            and leaf.store_param_remainders
            and not leaf.capturable
            and all(dtype == torch.float32 for dtype in leaf.name_to_dtype_map.values())
            and all(
                type(param) is torch.Tensor or type(param) is torch.nn.Parameter
                for group in leaf.param_groups
                for param in group["params"]
            )
            and all(param.dtype == torch.bfloat16 for group in leaf.param_groups for param in group["params"])
        )
    if not supported:
        yield False
        return
    handles = []
    try:
        for leaf in leaves:
            if leaf._optimizer_load_state_dict_pre_hooks or leaf._optimizer_load_state_dict_post_hooks:
                raise RuntimeError("Cannot stage TE restore with existing load-state hooks")
            handles.append(leaf.register_load_state_dict_pre_hook(_stage_te_state))
        yield True
    finally:
        for handle in handles:
            handle.remove()
