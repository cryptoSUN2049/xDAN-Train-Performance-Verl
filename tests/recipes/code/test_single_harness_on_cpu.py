"""CPU contracts for the one-arm DSH bring-up path."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize("mode", ["prompt", "step-hash", "paired-validation"])
def test_one_harness_routes_every_sample_to_that_arm(monkeypatch, tmp_path, mode):
    from recipes.code.mimoagent_runner import _select_config_path

    config = tmp_path / "agent.yaml"
    config.write_text("agent: {type: dsh-sdk}\n")
    spec = tmp_path / "single.yaml"
    spec.write_text("harnesses:\n  - label: dsh-sdk\n    config: ./agent.yaml\n")
    monkeypatch.setenv("MIXED_HARNESS_SPEC", str(spec))
    monkeypatch.setenv("MIXED_HARNESS_MODE", mode)
    monkeypatch.setenv("MIXED_HARNESS_SEED", "19")

    for index in range(4):
        for rollout_index in range(2):
            selected = _select_config_path(
                sample_index=index,
                tools_kwargs={"dataset_index": index, "rollout_index": rollout_index, "harness_round": index},
            )
            assert selected == (str(config), "dsh-sdk")


@pytest.mark.parametrize("entries", [[], {}, None])
def test_empty_or_non_list_harness_spec_is_rejected(monkeypatch, tmp_path, entries):
    from recipes.code.mimoagent_runner import _mixed_harness_specs

    spec = tmp_path / "invalid.yaml"
    spec.write_text(yaml.safe_dump({"harnesses": entries}))
    monkeypatch.setenv("MIXED_HARNESS_SPEC", str(spec))
    with pytest.raises(ValueError, match="at least one entry"):
        _mixed_harness_specs()


@pytest.mark.parametrize("gpu_count", [None, "1"])
@pytest.mark.parametrize("cpu_offload", [None, "False", "True", "fAlSe", "tRuE"])
def test_minimal_launcher_forwards_optimizer_choice_resume_and_public_route(tmp_path, gpu_count, cpu_offload):
    scripts = tmp_path / "scripts" / "code"
    scripts.mkdir(parents=True)
    launcher = scripts / "train-dsh-minimal.sh"
    shutil.copyfile(REPO_ROOT / "scripts/code/train-dsh-minimal.sh", launcher)
    capture = tmp_path / "capture.py"
    capture.write_text("import json, os, sys; print(json.dumps({'env': dict(os.environ), 'args': sys.argv[1:]}))")
    (scripts / "train.sh").write_text(
        f'#!/bin/bash\nexec {shlex.quote(sys.executable)} {shlex.quote(str(capture))} "$@"\n'
    )
    env = {
        "PATH": os.environ["PATH"],
        "MODEL_PATH": "/model",
        "TRAIN_DATA": "/train.parquet",
        "VAL_DATA": "/val.parquet",
        "DSH_GATEWAY_PUBLIC_ORIGIN": "https://gateway.example.test",
        "DSH_GATEWAY_ROUTE_DIR": "/tmp/routes with spaces",
        "MODAL_CONFIG_PATH": "/tmp/modal.toml",
        "MODAL_TOKEN_SECRET": "test-secret-not-for-cli",
    }
    if gpu_count is not None:
        env["TRAIN_NGPUS_PER_NODE"] = gpu_count
    if cpu_offload is not None:
        env["CPU_OPTIMIZER_OFFLOAD"] = cpu_offload
    resume = ["trainer.resume_mode=resume_path", "trainer.resume_from_path=/checkpoints/global_step_1"]
    result = subprocess.run(["bash", str(launcher), *resume], env=env, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    actual = data["env"]
    for key in ("TRAIN_NNODES", "ACTOR_PP", "ACTOR_CP", "ACTOR_EP"):
        assert actual[key] == "1"
    for key in ("TRAIN_NGPUS_PER_NODE", "ROLLOUT_NGPUS_PER_NODE", "ACTOR_TP", "ROLLOUT_TP"):
        assert actual[key] == (gpu_count or "2")
    for key in (
        "TRAIN_BATCH_SIZE",
        "PPO_MINI_BATCH_SIZE",
        "AGENT_NUM_WORKERS",
        "GATEWAY_COUNT",
        "MAX_CONCURRENT_SESSIONS",
    ):
        assert actual[key] == "1"
    assert actual["N"] == actual["TOTAL_STEPS"] == "2"
    assert actual["MAXLEN"] == "65536"
    assert actual["FILTER_GROUPS_ENABLE"].lower() == "false"
    assert actual["SAVE_FREQ"] == "1"
    assert actual["MIMOAGENT_HARNESS_SPEC"].endswith("/config/agent/code/dsh-only.yaml")
    args = data["args"]
    enabled = cpu_offload is not None and cpu_offload.lower() == "true"
    expected_flag = "True" if enabled else "False"
    expected_fraction = "1.0" if enabled else "0.0"
    assert actual["CPU_OPTIMIZER_OFFLOAD"] == expected_flag
    assert actual["MEGATRON_OFFLOAD"] == "True"
    assert f"+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload={expected_flag}" in args
    assert (
        f"+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction={expected_fraction}"
    ) in args
    assert "actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True" in args
    assert "++actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=True" in args
    assert not any("store_param_remainders=" in arg for arg in args)
    for field in ("main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        assert f"actor_rollout_ref.actor.optim.{field}=fp32" in args
    assert args[-2:] == resume
    for key in ("DSH_GATEWAY_PUBLIC_ORIGIN", "DSH_GATEWAY_ROUTE_DIR", "MODAL_CONFIG_PATH"):
        assert f'+ray_kwargs.ray_init.runtime_env.env_vars.{key}="{env[key]}"' in args
    assert not any("test-secret" in arg or "MODAL_TOKEN_SECRET" in arg for arg in args)


def test_minimal_launcher_explicit_precision_overrides_remain_last(tmp_path):
    scripts = tmp_path / "scripts" / "code"
    scripts.mkdir(parents=True)
    launcher = scripts / "train-dsh-minimal.sh"
    shutil.copyfile(REPO_ROOT / "scripts/code/train-dsh-minimal.sh", launcher)
    (scripts / "train.sh").write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    env = {
        "PATH": os.environ["PATH"],
        "DSH_GATEWAY_PUBLIC_ORIGIN": "https://gateway.example.test",
        "DSH_GATEWAY_ROUTE_DIR": "/tmp/routes",
    }
    explicit = [
        "actor_rollout_ref.actor.optim.use_precision_aware_optimizer=False",
        "++actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=False",
    ]
    result = subprocess.run(["bash", str(launcher), *explicit], env=env, check=True, capture_output=True, text=True)
    args = result.stdout.splitlines()
    assert "actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True" in args[:-2]
    assert "++actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=True" in args[:-2]
    assert args[-2:] == explicit


def test_dsh_specs_resolve_to_modal_agents(monkeypatch):
    from recipes.code.mimoagent_runner import _mixed_harness_specs

    config_dir = REPO_ROOT / "config/agent/code"
    for filename, labels in (("dsh-only.yaml", ["dsh-sdk"]), ("mix-dsh-mimocode.yaml", ["dsh-sdk", "mini-mimocode"])):
        monkeypatch.setenv("MIXED_HARNESS_SPEC", str(config_dir / filename))
        specs = _mixed_harness_specs()
        assert [label for label, _ in specs] == labels
        for _, path in specs:
            config = yaml.safe_load(Path(path).read_text())
            assert config["environment"]["environment_class"] == "modal"
            assert config["environment"]["anti_hack_cleanup"] is True
            assert config["environment"]["git_leak_prevention"] == "strip"
            assert config["environment"]["registry_secret"] == "mimo-dsh-ghcr-20260928"
            assert config["model"]["model_name"] == "policy"


@pytest.mark.parametrize("missing", ["DSH_GATEWAY_PUBLIC_ORIGIN", "DSH_GATEWAY_ROUTE_DIR"])
@pytest.mark.parametrize("unset", [True, False])
def test_minimal_launcher_rejects_missing_route_before_downstream_start(tmp_path, missing, unset):
    scripts = tmp_path / "scripts" / "code"
    scripts.mkdir(parents=True)
    launcher = scripts / "train-dsh-minimal.sh"
    shutil.copyfile(REPO_ROOT / "scripts/code/train-dsh-minimal.sh", launcher)
    marker = tmp_path / "downstream-started"
    (scripts / "train.sh").write_text(f"#!/bin/bash\ntouch {shlex.quote(str(marker))}\n")
    env = {
        "PATH": os.environ["PATH"],
        "DSH_GATEWAY_PUBLIC_ORIGIN": "https://gateway.example.test",
        "DSH_GATEWAY_ROUTE_DIR": "/tmp/routes",
    }
    if unset:
        del env[missing]
    else:
        env[missing] = ""
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert missing in result.stderr
    assert not marker.exists()


def test_minimal_launcher_does_not_require_modal_config_path(tmp_path):
    scripts = tmp_path / "scripts" / "code"
    scripts.mkdir(parents=True)
    launcher = scripts / "train-dsh-minimal.sh"
    shutil.copyfile(REPO_ROOT / "scripts/code/train-dsh-minimal.sh", launcher)
    (scripts / "train.sh").write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
    env = {
        "PATH": os.environ["PATH"],
        "DSH_GATEWAY_PUBLIC_ORIGIN": "https://gateway.example.test",
        "DSH_GATEWAY_ROUTE_DIR": "/tmp/routes",
    }
    result = subprocess.run(["bash", str(launcher)], env=env, check=True, capture_output=True, text=True)
    assert "DSH_GATEWAY_PUBLIC_ORIGIN" in result.stdout
    assert "MODAL_CONFIG_PATH" not in result.stdout


@pytest.mark.parametrize("value", ["", "0", "1", "yes", "invalid"])
def test_minimal_launcher_rejects_invalid_cpu_optimizer_switch_before_start(tmp_path, value):
    scripts = tmp_path / "scripts" / "code"
    scripts.mkdir(parents=True)
    launcher = scripts / "train-dsh-minimal.sh"
    shutil.copyfile(REPO_ROOT / "scripts/code/train-dsh-minimal.sh", launcher)
    marker = tmp_path / "downstream-started"
    (scripts / "train.sh").write_text(f"#!/bin/bash\ntouch {shlex.quote(str(marker))}\n")
    env = {
        "PATH": os.environ["PATH"],
        "DSH_GATEWAY_PUBLIC_ORIGIN": "https://gateway.example.test",
        "DSH_GATEWAY_ROUTE_DIR": "/tmp/routes",
        "CPU_OPTIMIZER_OFFLOAD": value,
    }
    result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "CPU_OPTIMIZER_OFFLOAD" in result.stderr
    assert not marker.exists()
