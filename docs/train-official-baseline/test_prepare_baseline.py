"""CPU checks for an unchanged official source and its small native Code run."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VENDOR = REPO.parent / "train-p0-integration"
SPEC = importlib.util.spec_from_file_location("prepare_baseline", HERE / "prepare_baseline.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path / "prepared"
    m.prepare(REPO, VENDOR, output)
    return output


def test_native_profiles_preserve_agent_and_official_request_defaults(prepared):
    for name in m.ARMS:
        original = yaml.safe_load(m.source_bytes(REPO, f"config/agent/code/{name}.yaml"))
        actual = yaml.safe_load((prepared / "harness" / f"{name}.yaml").read_text())
        assert (prepared / "harness" / f"{name}.yaml").read_bytes().split(b"environment:\n")[0] == m.source_bytes(
            REPO, f"config/agent/code/{name}.yaml"
        ).split(b"environment:\n")[0]
        assert actual["agent"] == original["agent"]
        expected_model = original["model"]
        key = "max_output_tokens" if name == "mini-codex" else "max_tokens"
        expected_model["model_kwargs"].update({key: 32768, "timeout": 3600, "max_retries": 0})
        assert actual["model"] == expected_model
        assert actual["environment"]["environment_class"] == "modal"
        assert actual["environment"]["registry_secret"] == "mimo-dsh-ghcr-20260928"
        assert actual["environment"]["cpu"] == 2
        assert actual["environment"]["memory"] == 8192
        assert actual["environment"]["anti_hack_cleanup"] is True
    assert (prepared / "harness" / "mix-four-whitebox.yaml").read_bytes() == m.source_bytes(
        REPO, "config/agent/code/mix-four-whitebox.yaml"
    )


def test_source_archive_contains_exact_official_trainer_and_pinned_vendors(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text())
    with tarfile.open(prepared / "source.tar.gz") as archive:
        files = [entry for entry in archive if entry.isfile()]
        assert len(files) == manifest["source_file_count"]
        assert {entry.name: entry.linkname for entry in archive.getmembers() if entry.issym()} == manifest[
            "source_symlinks"
        ]
        assert len(manifest["source_git_modes"]) == len(files) + manifest["source_symlink_count"]
        assert manifest["source_git_modes"]["CLAUDE.md"] == "120000"
        assert (
            manifest["source_git_modes"]["scripts/code/train.sh"]
            == m.git_modes(REPO, m.COMMIT)["scripts/code/train.sh"]
        )
        for path in ("recipes/code/mimoagent_runner.py", "verl/trainer/ppo/v1/trainer_base.py"):
            assert archive.extractfile(path).read() == m.source_bytes(REPO, path)
        vendor_path = "third_party/mimoagent-osr/src/mimoagent/environments/datasets/opensource_code.py"
        assert archive.extractfile(vendor_path).read() == subprocess.check_output(
            [
                "git",
                "-C",
                str(VENDOR / "third_party/mimoagent-osr"),
                "show",
                f"{m.PINS['third_party/mimoagent-osr']}:src/mimoagent/environments/datasets/opensource_code.py",
            ]
        )
        assert not any("dsh_agent" in entry.name or "complete_group_sampler" in entry.name for entry in files)
    assert manifest["source_commit"] == m.COMMIT
    assert manifest["submodules"] == m.PINS
    assert hashlib.sha256((prepared / "source.tar.gz").read_bytes()).hexdigest() == manifest["source_archive_sha256"]
    assert manifest["cpu_config_preflight"] == "pending"
    assert manifest["deadline_utc"] == "2026-09-30T12:51:43.743Z"


def test_launch_separates_fresh_and_real_resume_without_custom_algorithms(prepared):
    launch = (prepared / "launch.sh").read_text()
    subprocess.run(["bash", "-n", str(prepared / "launch.sh")], check=True)
    assert "scripts/code/train.sh" in launch
    assert "TRAINER_MODE=colocate_async" in launch
    assert "ACTOR_TP=4" in launch
    assert "ROLLOUT_TP=2" in launch
    assert "MAXLEN=65536" in launch
    assert "PROMPT_LENGTH=4096" in launch
    assert "RESPONSE_LENGTH=61440" in launch
    assert "sglang.context_length=65536" in launch
    assert "trainer.resume_mode=resume_path" in launch
    assert "global_step_1" in launch
    assert "a9off001" in launch and "a9off002" in launch
    assert "MIXED_HARNESS_MODE=step-hash" in launch
    assert "ALGORITHM_GROUP_ADVANTAGE_BY_HARNESS=False" in launch
    assert "warmup_grad_sync" not in launch
    assert "custom_sampler" not in launch
    assert "WANDB_API_KEY=" not in launch
    assert "MODAL_TOKEN_SECRET=" not in launch
    assert 'export PYTHONPATH="$SOURCE:' in launch
    assert "RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080" in launch
    assert "VERL_RL_INSIGHT_ENABLE=1" in launch
    assert "MIMOAGENT_RG_PATH=/workspace/train-p0-dsh-integration/tools/mimoagent-cache/ripgrep/15.1.0/rg" in launch
    assert "trainer.logger=[console,tensorboard,file,wandb,rl_insight]" in launch


def test_existing_output_is_never_overwritten(prepared):
    before = (prepared / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        m.prepare(REPO, VENDOR, prepared)
    assert (prepared / "manifest.json").read_bytes() == before


def test_runtime_forwarding_keeps_paths_as_hydra_strings():
    line = next(line for line in m.launch_bytes().decode().splitlines() if "FORWARD_ARGS+=(" in line)
    probe = (
        "name=NETRC; NETRC=/root/private/wandb.netrc; FORWARD_ARGS=();\n" + line + '\nprintf "%s" "${FORWARD_ARGS[0]}"'
    )
    actual = subprocess.check_output(["bash", "-c", probe], text=True)
    assert actual == '+ray_kwargs.ray_init.runtime_env.env_vars.NETRC="/root/private/wandb.netrc"'


def test_wrong_pin_fails_before_output_write(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "PINS", {**m.PINS, "third_party/mimoagent-osr": "0" * 40})
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="gitlink"):
        m.prepare(REPO, VENDOR, output)
    assert not output.exists()


@pytest.mark.parametrize("phase,steps,resume", [("fresh", 1, "disable"), ("resume", 2, "resume_path")])
def test_real_cpu_hydra_config_and_official_validator(tmp_path, phase, steps, resume):
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    run = m.RUN if phase == "fresh" else m.RUN + "/resume-step2"
    values = {
        "actor_rollout_ref.model.path": "/workspace/models/MiMo-V2.6-Distill-Qwen-9B",
        "data.train_files": f"[{m.BASE}/data-train8-is1-r1/train.parquet]",
        "data.val_files": f"[{m.BASE}/data-minimal-r1/holdout.parquet]",
        "data.train_batch_size": 1,
        "data.val_batch_size": 1,
        "data.max_prompt_length": 4096,
        "data.max_response_length": 61440,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 1,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.calculate_entropy": True,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.actor.megatron.tensor_model_parallel_size": 4,
        "actor_rollout_ref.actor.megatron.context_parallel_size": 1,
        "actor_rollout_ref.ref.megatron.tensor_model_parallel_size": 4,
        "actor_rollout_ref.ref.megatron.context_parallel_size": 1,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 2,
        "actor_rollout_ref.rollout.nnodes": 0,
        "actor_rollout_ref.rollout.n_gpus_per_node": 4,
        "actor_rollout_ref.rollout.n": 4,
        "actor_rollout_ref.rollout.max_model_len": 65536,
        "actor_rollout_ref.rollout.prompt_length": 4096,
        "actor_rollout_ref.rollout.response_length": 61440,
        "actor_rollout_ref.rollout.custom.agent_framework.repetition_detect.enable": True,
        "++actor_rollout_ref.rollout.engine_kwargs.sglang.context_length": 65536,
        "++actor_rollout_ref.rollout.engine_kwargs.sglang.max_mamba_cache_size": 384,
        "trainer.v1.trainer_mode": "colocate_async",
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": 4,
        "trainer.total_training_steps": steps,
        "trainer.save_freq": 1,
        "trainer.test_freq": 2,
        "trainer.logger": "[console,tensorboard,file,wandb,rl_insight]",
        "trainer.default_local_dir": m.CHECKPOINT,
        "trainer.rollout_data_dir": run + "/rollouts",
        "trainer.validation_data_dir": run + "/validation",
        "trainer.resume_mode": resume,
        "trainer.resume_from_path": "null" if phase == "fresh" else m.CHECKPOINT + "/global_step_1",
        "transfer_queue.enable": True,
    }
    # Use the exact official YAMLs via a filesystem search path so this CPU test
    # needs neither torch nor verl's runtime imports. No config is instantiated.
    overrides = [f"hydra.searchpath=[file://{REPO}/verl/trainer/config]"]
    overrides.extend(f"{key}={value}" for key, value in values.items())
    with hydra.initialize_config_dir(config_dir=str(REPO / "recipes/code/config"), version_base=None):
        cfg = hydra.compose(config_name="train", overrides=overrides)
        resolved = tmp_path / "resolved.yaml"
        resolved.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    assert cfg.actor_rollout_ref.actor.optim.main_grads_dtype == "fp32"
    assert cfg.actor_rollout_ref.actor.optim.exp_avg_dtype == "fp32"
    assert cfg.actor_rollout_ref.actor.optim.exp_avg_sq_dtype == "fp32"
    assert cfg.actor_rollout_ref.actor.optim.use_precision_aware_optimizer is False
    assert cfg.algorithm.group_advantage_by_harness is False
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.sglang.kv_cache_dtype == "fp8_e4m3"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "recipes/code/validate_resolved_config.py"),
            str(resolved),
            "--save-freq",
            "1",
            "--checkpoint-dir",
            m.CHECKPOINT,
            "--rollout-data-dir",
            run + "/rollouts",
            "--validation-data-dir",
            run + "/validation",
            "--temperature",
            "1",
            "--top-p",
            "0.95",
            "--top-k",
            "20",
            "--mamba-cache-size",
            "384",
            "--mamba-scheduler",
            "extra_buffer",
            "--entropy-coeff",
            "0",
            "--filter-groups-enabled",
            "true",
            "--tool-call-error-penalty-enabled",
            "false",
            "--tool-call-error-penalty-strategy",
            "monitor",
            "--tool-call-error-penalty-value",
            "0",
            "--repetition-detect-enabled",
            "true",
            "--repetition-zero-reward",
            "false",
            "--repetition-penalty-enabled",
            "false",
            "--repetition-penalty-strategy",
            "monitor",
            "--repetition-penalty-value",
            "0",
        ],
        check=True,
    )
