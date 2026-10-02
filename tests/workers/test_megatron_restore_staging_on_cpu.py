"""Execute production restore/context methods without importing a CUDA stack.

AST extraction only avoids optional Megatron imports; assertions observe buffer
lifetimes and checkpoint state across restore followed by a train context.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]


def extract(path, class_name, method=None, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    if method is not None:
        node = next(node for node in node.body if isinstance(node, ast.FunctionDef) and node.name == method)
    module = ast.Module(body=[node], type_ignores=[])
    scope = dict(namespace or {})
    exec(compile(module, str(ROOT / path), "exec"), scope)
    return scope[method or class_name]


@pytest.mark.parametrize(
    "cpu_optimizer,fsdp,expected_grad", [(False, False, False), (True, False, True), (False, True, True)]
)
def test_restore_defers_only_gpu_optimizer_gradients_and_train_reallocates(cpu_optimizer, fsdp, expected_grad):
    state = {"model_resident": False, "grad_resident": False, "optimizer_step": None, "calls": []}

    def load_model(module, load_grad=True):
        state.update(model_resident=True, grad_resident=load_grad)
        state["calls"].append(("load", load_grad))

    def offload_model(module):
        state.update(model_resident=False, grad_resident=False)
        state["calls"].append(("offload",))

    def restore(**kwargs):
        assert state["model_resident"]
        assert state["grad_resident"] is expected_grad
        assert kwargs["local_path"] == "step2"
        assert kwargs["del_local_after_load"] is False
        state["optimizer_step"] = 2
        state["calls"].append(("restore",))

    engine = SimpleNamespace(
        _is_offload_param=True,
        _is_offload_optimizer=True,
        is_param_offload_enabled=True,
        is_optimizer_offload_enabled=True,
        optimizer_config=SimpleNamespace(override_optimizer_config={"optimizer_cpu_offload": cpu_optimizer}),
        engine_config=SimpleNamespace(use_megatron_fsdp=fsdp),
        module=object(),
        optimizer=object(),
        checkpoint_mananager=SimpleNamespace(load_checkpoint=restore),
    )
    load = extract(
        "verl/workers/engine/megatron/transformer_impl.py",
        "MegatronEngine",
        "load_checkpoint",
        {
            "Optional": Optional,
            "load_megatron_model_to_gpu": load_model,
            "offload_megatron_model_to_cpu": offload_model,
            "offload_megatron_optimizer": lambda optimizer: state["calls"].append(("offload_optimizer",)),
        },
    )
    load(engine, "step2", del_local_after_load=False)
    assert state["calls"] == [("load", expected_grad), ("restore",), ("offload",), ("offload_optimizer",)]
    assert state["optimizer_step"] == 2 and not state["grad_resident"]

    def move(device, model=True, optimizer=True, grad=True):
        if model:
            if device == "cuda":
                load_model(engine.module, load_grad=grad)
            else:
                offload_model(engine.module)

    engine.to = move
    context = extract(
        "verl/workers/engine/base.py",
        "BaseEngineCtx",
        namespace={"BaseEngine": object, "get_device_name": lambda: "cuda"},
    )
    with context(engine, mode="train"):
        assert state["grad_resident"] and state["model_resident"]
        assert state["optimizer_step"] == 2
    assert not state["grad_resident"] and engine.mode is None


def test_without_parameter_offload_no_extra_load_or_configuration_access():
    calls = []
    engine = SimpleNamespace(
        _is_offload_param=False,
        _is_offload_optimizer=False,
        checkpoint_mananager=SimpleNamespace(load_checkpoint=lambda **kwargs: calls.append(kwargs)),
    )
    load = extract(
        "verl/workers/engine/megatron/transformer_impl.py", "MegatronEngine", "load_checkpoint", {"Optional": Optional}
    )
    load(engine, "step2", del_local_after_load=False)
    assert calls == [{"local_path": "step2", "hdfs_path": None, "del_local_after_load": False}]
