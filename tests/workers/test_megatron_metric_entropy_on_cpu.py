"""Execute production functions with CPU autograd and single-rank TP collectives.

AST loading avoids importing CUDA-only Megatron. Only the compile decorator and
collective transport are stubbed; entropy forward/backward are production code.
"""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import NonTensorData, TensorDict

ROOT = Path(__file__).resolve().parents[2]


def load(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names]
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                child.decorator_list = [
                    d for d in child.decorator_list if isinstance(d, ast.Name) and d.id == "staticmethod"
                ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)
    return namespace


def processor(chunked):
    ns = {
        "torch": torch,
        "nullcontext": nullcontext,
        "dist": SimpleNamespace(all_reduce=lambda *a, **k: None, ReduceOp=SimpleNamespace(MAX=0)),
        "mpu": SimpleNamespace(get_tensor_model_parallel_group=lambda: None),
    }
    load(
        "verl/utils/megatron/tensor_parallel.py",
        {"_VocabParallelEntropy", "vocab_parallel_entropy", "vocab_parallel_entropy_with_chunking"},
        ns,
    )
    pointers = []

    def log_probs(logits, labels):
        pointers.append(logits.data_ptr())
        return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    ns["vocab_parallel_log_probs_from_logits"] = log_probs
    load("verl/workers/engine/megatron/transformer_impl.py", {"_lm_head_logits_processor"}, ns)
    engine = SimpleNamespace(
        engine_config=SimpleNamespace(entropy_from_logits_with_chunking=chunked, entropy_from_logits_chunk_size=2)
    )
    return ns["_lm_head_logits_processor"], engine, pointers


class Data(dict):
    def select(self, *keys):
        return Data({k: self[k] for k in keys})

    def to_padded_tensor(self):
        return self


class Metric(SimpleNamespace):
    @classmethod
    def from_dict(cls, values, aggregation):
        return {k: cls(value=v, aggregation=aggregation) for k, v in values.items()}


def production_loss(output, coeff):
    # Policy objective is a controlled differentiable fixture; production PPO
    # dispatch, entropy aggregation, coefficient handling and metrics execute.
    def policy(**kwargs):
        return -(kwargs["log_prob"] * kwargs["advantages"]).mean(), {}

    ns = {
        "no_padding_2_padding": lambda x, _: x,
        "Metric": Metric,
        "AggregationType": SimpleNamespace(SUM="sum", MEAN="mean"),
        "get_policy_loss_fn": lambda _: policy,
        "agg_loss": lambda loss_mat, **_: loss_mat.mean(),
    }
    load("verl/workers/utils/losses.py", {"ppo_loss"}, ns)
    config = SimpleNamespace(
        global_batch_info={},
        loss_scale_factor=None,
        loss_agg_mode="token-mean",
        policy_loss={},
        entropy_coeff=coeff,
        use_kl_loss=False,
    )
    data = Data(
        dp_size=1,
        batch_num_tokens=None,
        global_batch_size=None,
        response_mask=torch.ones(1, 4),
        old_log_probs=torch.zeros(1, 4),
        advantages=torch.tensor([[0.75, -0.25, 0.75, -0.25]]),
    )
    return ns["ppo_loss"](config, output, data)


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("coeff", [0.0, 0.01, -0.01])
def test_entropy_value_gradient_and_input_contract(chunked, coeff):
    fn, engine, pointers = processor(chunked)
    if coeff != 0.0:
        # Multiple gradient-bearing views have an existing kernel version-counter
        # failure; preserve that behavior separately and test this contract with one chunk.
        engine.engine_config.entropy_from_logits_chunk_size = 8
    original = torch.linspace(-1.3, 2.1, 28).reshape(1, 4, 7)
    results = []
    for requires_grad in (True, coeff != 0.0):
        leaf = original.clone().requires_grad_()
        logits = leaf * 1.0
        output = fn(
            engine,
            logits,
            torch.tensor([[0, 2, 4, 6]]),
            torch.full((1, 4), 0.7),
            calculate_sum_pi_squared=False,
            calculate_entropy=True,
            distillation_use_topk=False,
            distillation_only=False,
            logits_processor_func=None,
            batch=None,
            data_format="bshd",
            entropy_requires_grad=requires_grad,
        )
        torch.testing.assert_close(logits, original / 0.7)
        assert (pointers[-1] == logits.data_ptr()) is (not requires_grad)
        assert output["entropy"].requires_grad is requires_grad
        loss, metrics = production_loss(output, coeff)
        # Baseline is the former unconditional entropy multiplication.
        baseline = (
            -(output["log_probs"] * torch.tensor([[0.75, -0.25, 0.75, -0.25]])).mean()
            - coeff * output["entropy"].mean()
        )
        torch.testing.assert_close(loss, baseline)
        (baseline if requires_grad and not chunked else loss).backward()
        torch.testing.assert_close(logits, original / 0.7)
        results.append((loss.detach(), leaf.grad, metrics["actor/entropy_loss"].value.detach()))
    for a, b in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(a, b)
    # Independent dense softmax oracle proves nonzero entropy gradient survives.
    reference = original.clone().requires_grad_()
    logp = (reference / 0.7).log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    expected = (
        -(
            logp.gather(-1, torch.tensor([[[0], [2], [4], [6]]])).squeeze(-1)
            * torch.tensor([[0.75, -0.25, 0.75, -0.25]])
        ).mean()
        - coeff * entropy.mean()
    )
    expected.backward()
    torch.testing.assert_close(results[1][1], reference.grad)
    torch.testing.assert_close(results[1][2], entropy.mean())


def test_unknown_caller_preserves_entropy_gradient_and_clone():
    fn, engine, pointers = processor(False)
    logits = torch.randn(1, 4, 7, requires_grad=True) * 1
    output = fn(
        engine,
        logits,
        torch.zeros(1, 4, dtype=torch.long),
        torch.ones(1, 4),
        calculate_sum_pi_squared=False,
        calculate_entropy=True,
        distillation_use_topk=False,
        distillation_only=False,
        logits_processor_func=None,
        batch=None,
        data_format="bshd",
    )
    assert output["entropy"].requires_grad
    assert pointers[-1] != logits.data_ptr()


def test_zero_coefficient_does_not_connect_entropy_to_loss():
    entropy = torch.randn(1, 4, requires_grad=True)
    log_probs = torch.randn(1, 4, requires_grad=True)
    loss, metrics = production_loss({"log_probs": log_probs, "entropy": entropy}, 0.0)
    loss.backward()
    assert entropy.grad is None
    assert log_probs.grad is not None
    torch.testing.assert_close(metrics["actor/entropy_loss"].value, entropy.mean())


@pytest.mark.parametrize("coeff", [0.0, 0.01])
def test_actor_metadata_survives_actual_static_microbatch_split(coeff):
    ns = {"is_distillation_enabled": lambda _: False, "rename_dict": lambda x, _: x, "reduce_metrics": lambda x: x}
    load("verl/trainer/ppo/v1/trainer_base.py", {"_update_actor"}, ns)
    actor = SimpleNamespace(
        ppo_mini_batch_size=2,
        loss_agg_mode="token-mean",
        calculate_entropy=True,
        entropy_coeff=coeff,
        ppo_epochs=1,
        data_loader_seed=0,
        shuffle=False,
    )
    config = Data(actor_rollout_ref=SimpleNamespace(actor=actor, rollout=SimpleNamespace(n=1, temperature=1.0)))
    config.actor_rollout_ref = config["actor_rollout_ref"]
    batch = SimpleNamespace(extra_info={})
    trainer = SimpleNamespace(
        config=config, actor_rollout_wg=SimpleNamespace(update_actor=lambda _: {"metrics": {"actor/mfu": 0.0}})
    )
    ns["_update_actor"](trainer, batch, {})
    assert batch.extra_info["calculate_entropy"] is True
    td_ns = {
        "torch": torch,
        "TensorDict": TensorDict,
        "unwrap_non_tensor_data": lambda x: x.data if isinstance(x, NonTensorData) else x,
    }
    load("verl/utils/tensordict_utils.py", {"get_non_tensor_data", "chunk_tensordict"}, td_ns)
    tu = SimpleNamespace(**{k: td_ns[k] for k in ("get_non_tensor_data", "chunk_tensordict")})
    engine_ns = load("verl/workers/engine/utils.py", {"prepare_micro_batches"}, {"tu": tu})
    data = TensorDict(
        {
            "input_ids": torch.ones(2, 4),
            "use_dynamic_bsz": NonTensorData(False),
            "micro_batch_size_per_gpu": NonTensorData(1),
            "entropy_requires_grad": NonTensorData(batch.extra_info["entropy_requires_grad"]),
        },
        batch_size=[2],
    )
    micro_batches, _ = engine_ns["prepare_micro_batches"](data)
    assert len(micro_batches) == 2
    for micro in micro_batches:
        assert tu.get_non_tensor_data(micro, "entropy_requires_grad", True) is (coeff != 0.0)


def test_existing_multichunk_gradient_version_counter_failure_is_unchanged():
    fn, engine, _ = processor(True)
    logits = torch.randn(1, 4, 7, requires_grad=True) * 1
    output = fn(
        engine,
        logits,
        torch.zeros(1, 4, dtype=torch.long),
        torch.ones(1, 4),
        calculate_sum_pi_squared=False,
        calculate_entropy=True,
        distillation_use_topk=False,
        distillation_only=False,
        logits_processor_func=None,
        batch=None,
        data_format="bshd",
    )
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output["entropy"].sum().backward()


@pytest.mark.parametrize("chunked", [False, True])
def test_metric_entropy_saves_no_autograd_tensors(chunked):
    fn, engine, _ = processor(chunked)
    counts = []
    for requires_grad in (True, False):
        saved = []
        logits = torch.randn(1, 4, 7, requires_grad=True) * 1
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor, saved=saved: saved.append(tensor) or tensor, lambda tensor: tensor
        ):
            output = fn(
                engine,
                logits,
                torch.zeros(1, 4, dtype=torch.long),
                torch.ones(1, 4),
                calculate_sum_pi_squared=False,
                calculate_entropy=True,
                distillation_use_topk=False,
                distillation_only=True,
                logits_processor_func=None,
                batch=None,
                data_format="bshd",
                entropy_requires_grad=requires_grad,
            )
        counts.append(len(saved))
        assert output["entropy"].requires_grad is requires_grad
    # Temperature division itself saves one tensor in both paths.
    assert counts[0] > counts[1]
    assert counts[1] == 1
