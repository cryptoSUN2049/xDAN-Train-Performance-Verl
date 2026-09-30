"""Prepare the pinned official Code recipe; never deploy, create a sandbox, or train."""

from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

import yaml

COMMIT = "a2ad9f6160b03ff2d47e59832bfb6b289f37c917"
PINS = {
    "third_party/mimoagent-osr": "467f0a19016f0ac4d63b8d17a1f0da9ba07f232c",
    "third_party/uni_agent": "c63e0b01c375ebede95e01fe92bc367df24e5bf3",
}
ARMS = ("mini-mimocode", "mini-bash", "mini-claude-code", "mini-codex")
BASE = "/workspace/train-p0-dsh-integration"
SOURCE = BASE + "/source-a2ad9f61"
RUN_ID = "official-code-4gpu-64k-r1-20260930"
RUN = BASE + "/runs/" + RUN_ID
CHECKPOINT = "/opt/train-p0-dsh-integration/checkpoints/" + RUN_ID
VENV = "/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/venv"
DEADLINE = "2026-09-30T12:51:43.743Z"


def git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def source_bytes(repo: Path, path: str) -> bytes:
    return git_bytes(repo, "show", f"{COMMIT}:{path}")


def git_modes(repo: Path, commit: str, prefix: str = "") -> dict[str, str]:
    modes = {}
    for record in git_bytes(repo, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if record:
            metadata, name = record.split(b"\t", 1)
            mode, kind, _ = metadata.split()
            if kind == b"blob":
                modes[prefix + name.decode()] = mode.decode()
    return modes


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def modal_profile(original: bytes, name: str) -> bytes:
    config = yaml.safe_load(original)
    environment = config["environment"]
    for key in (
        "node_selector",
        "cpu_request",
        "memory_request",
        "cpu_limit",
        "memory_limit",
        "labels",
        "host_network",
    ):
        environment.pop(key, None)
    environment.update(
        environment_class="modal",
        cwd="/testbed",
        cpu=2,
        memory=8192,
        registry_secret="mimo-dsh-ghcr-20260928",
        raise_on_transport_error=True,
        sandbox_timeout=7200,
    )
    # Custom specs bypass train.sh's default-spec copy branch. Repeat its exact
    # request settings, including the Responses API key for the native codex arm.
    key = "max_output_tokens" if name == "mini-codex" else "max_tokens"
    text = original.decode()
    before_environment, environment_and_model = text.split("environment:\n", 1)
    _, model = environment_and_model.split("\nmodel:\n", 1)
    model = "model:\n" + model
    model = model.replace(
        "  model_kwargs:\n",
        f"  model_kwargs:\n    {key}: 32768\n    timeout: 3600\n    max_retries: 0\n",
        1,
    )
    environment_text = yaml.safe_dump({"environment": environment}, sort_keys=False, allow_unicode=True)
    return (before_environment + environment_text + "\n" + model).encode()


def launch_bytes() -> bytes:
    environment = {
        "UV_ENV_DIR": VENV,
        "SOURCE": SOURCE,
        "BASE_RUN": RUN,
        "RUN_ID": RUN_ID,
        "CHECKPOINT_DIR": CHECKPOINT,
        "MODEL_PATH": "/workspace/models/MiMo-V2.6-Distill-Qwen-9B",
        "TRAIN_DATA": BASE + "/data-train8-is1-r1/train.parquet",
        "VAL_DATA": BASE + "/data-minimal-r1/holdout.parquet",
        "NETRC": "/root/mimo-private/wandb.netrc",
        "MODAL_CONFIG_PATH": "/root/mimo-private/modal.toml",
        "MODAL_PROFILE": "l98348740",
        "RL_INSIGHT_SERVER_URL": "http://127.0.0.1:18080",
        "VERL_RL_INSIGHT_ENABLE": "1",
        "MIMOAGENT_RG_PATH": BASE + "/tools/mimoagent-cache/ripgrep/15.1.0/rg",
        "RAY_INIT_ADDRESS": "127.0.0.1:6381",
        "PROJECT_NAME": "xDAN-Train-Performance-Verl",
        "WANDB_ENTITY": "xdan-ai",
        "WANDB_MODE": "online",
        "WANDB_RESUME": "never",
        "TRAINER_MODE": "colocate_async",
        "TRAIN_NNODES": "1",
        "TRAIN_NGPUS_PER_NODE": "4",
        "ROLLOUT_NNODES": "0",
        "ROLLOUT_NGPUS_PER_NODE": "4",
        "ACTOR_TP": "4",
        "ACTOR_PP": "1",
        "ACTOR_CP": "1",
        "ACTOR_EP": "1",
        "ROLLOUT_TP": "2",
        "MAXLEN": "65536",
        "PROMPT_LENGTH": "4096",
        "RESPONSE_LENGTH": "61440",
        "PPO_MAX_TOKEN_LEN_PER_GPU": "65536",
        "N": "4",
        "TRAIN_BATCH_SIZE": "1",
        "PPO_MINI_BATCH_SIZE": "1",
        "MICRO_BSZ_PER_GPU": "1",
        "USE_DYNAMIC_BSZ": "False",
        "MEGATRON_OFFLOAD": "True",
        "SAVE_FREQ": "1",
        "TEST_FREQ": "2",
        "TOTAL_EPOCHS": "10",
        "VAL_N": "1",
        "VAL_BATCH_SIZE": "1",
        "VAL_DO_SAMPLE": "False",
        "AGENT_NUM_WORKERS": "1",
        "GATEWAY_COUNT": "1",
        "MAX_CONCURRENT_SESSIONS": "4",
        "ROLLOUT_MAX_RUNNING_REQUESTS": "4",
        "MIXED_HARNESS_MODE": "step-hash",
        "MIXED_HARNESS_SEED": "20260911",
        "ALGORITHM_GROUP_ADVANTAGE_BY_HARNESS": "False",
        "LOSS_AGG_MODE": "prompt-mean",
        "NORM_ADV_BY_STD_IN_GRPO": "False",
        "FILTER_GROUPS_ENABLE": "True",
        "FILTER_GROUPS_METRIC": "reward",
        "ENTROPY_COEFF": "0",
        "ENTROPY_CHUNKING": "True",
        "ENTROPY_CHUNK_SIZE": "16384",
        "MAX_OFF_POLICY_THRESHOLD": "2",
        "MAX_OFF_POLICY_STRATEGY": "drop",
        "REPETITION_DETECT_ENABLE": "true",
        "REPETITION_ZERO_REWARD": "false",
        "REPETITION_PENALTY_ENABLE": "false",
        "TOOL_CALL_ERROR_PENALTY_ENABLE": "false",
        "DEEP_FAILURE_MASK_ENABLE": "false",
        "HARNESS_TURN_MAX_TOKENS": "32768",
        "MODEL_REQUEST_TIMEOUT": "3600",
        "MODEL_SDK_MAX_RETRIES": "0",
    }
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"# Official source {COMMIT}; immutable deadline {DEADLINE}."]
    lines += [f"export {key}={shlex.quote(value)}" for key, value in environment.items()]
    lines += [
        'export PATH="$UV_ENV_DIR/bin:$PATH"',
        'test "$(command -v python3)" = "$UV_ENV_DIR/bin/python3"',
        'export PYTHONPATH="$SOURCE:$SOURCE/third_party/mimoagent-osr/src:$SOURCE/third_party/uni_agent"',
        'export MIMOAGENT_HARNESS_SPEC="$BASE_RUN/harness/mix-four-whitebox.yaml"',
        "unset WANDB_API_KEY MODAL_TOKEN_ID MODAL_TOKEN_SECRET",
        'PHASE="${1:-fresh}"',
        'test "$#" -le 1',
        "RESUME_ARGS=(trainer.resume_mode=disable trainer.resume_from_path=null)",
        'case "$PHASE" in',
        '  fresh) export TOTAL_STEPS=1 WANDB_RUN_ID=a9off001 RUN_DIR="$BASE_RUN" ;;',
        '  resume) export TOTAL_STEPS=2 WANDB_RUN_ID=a9off002 RUN_DIR="$BASE_RUN/resume-step2"',
        '    RESUME_ARGS=(trainer.resume_mode=resume_path "trainer.resume_from_path=$CHECKPOINT_DIR/global_step_1") ;;',
        '  *) echo "usage: launch.sh [fresh|resume]" >&2; exit 2 ;;',
        "esac",
        'export EXP_NAME="$RUN_ID-$PHASE" WANDB_DIR="$RUN_DIR"',
        'export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"',
        'export TENSORBOARD_DIR="$RUN_DIR/tensorboard" AGENT_DEBUG_DIR="$RUN_DIR/dumps"',
        'export UNI_AGENT_LOG_DIR="$RUN_DIR/trajectories" ROLLOUT_DATA_DIR="$RUN_DIR/rollouts"',
        'export VALIDATION_DATA_DIR="$RUN_DIR/validation" RESOLVED_CONFIG_PATH="$RUN_DIR/resolved_config.yaml"',
        'if [[ "${CPU_CONFIG_PREFLIGHT:-0}" == 1 ]]; then',
        '  export CUDA_VISIBLE_DEVICES="" PREFLIGHT_ONLY=1 SKIP_CLUSTER_CHECK=1',
        '  export RESOLVED_CONFIG_PATH="$RUN_DIR/cpu-resolved-config.yaml"',
        "else",
        '  test "${SKIP_CLUSTER_CHECK:-0}" != 1',
        '  test -f "$MODEL_PATH/config.json"',
        '  if [[ "$PHASE" == resume ]]; then',
        '    test -d "$CHECKPOINT_DIR/global_step_1/actor"',
        '    test -f "$CHECKPOINT_DIR/global_step_1/data.pt"',
        '  else test ! -e "$CHECKPOINT_DIR/global_step_1"; fi',
        "fi",
        'cd "$SOURCE"',
        "# The root operator must verify deadline guard and full checkpoint receipts before fit.",
        "FORWARD_ARGS=()",
        "for name in NETRC MODAL_CONFIG_PATH MODAL_PROFILE WANDB_ENTITY WANDB_RUN_ID "
        "WANDB_MODE WANDB_RESUME WANDB_DIR RUN_DIR VERL_FILE_LOGGER_PATH "
        "RL_INSIGHT_SERVER_URL VERL_RL_INSIGHT_ENABLE MIMOAGENT_RG_PATH; do",
        '  FORWARD_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.$name=\\"${!name}\\"")',
        "done",
        "exec bash scripts/code/train.sh \\",
        "  ++actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=65536 \\",
        "  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \\",
        "  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \\",
        "  actor_rollout_ref.actor.checkpoint.async_save=False \\",
        "  'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]' \\",
        "  'actor_rollout_ref.actor.checkpoint.load_contents=[model,hf_model,optimizer,extra]' \\",
        "  'trainer.logger=[console,tensorboard,file,wandb,rl_insight]' \\",
        '  "${FORWARD_ARGS[@]}" "${RESUME_ARGS[@]}"',
        "",
    ]
    return "\n".join(lines).encode()


def prepare(repo: Path, vendor_root: Path, output: Path) -> dict:
    archives = [("", git_bytes(repo, "archive", "--format=tar", COMMIT))]
    modes = git_modes(repo, COMMIT)
    for path, commit in PINS.items():
        tree_entry = git_bytes(repo, "ls-tree", COMMIT, path).decode().split()
        if len(tree_entry) < 3 or tree_entry[2] != commit:
            raise ValueError(f"official gitlink mismatch: {path}")
        archives.append((path + "/", git_bytes(vendor_root / path, "archive", "--format=tar", commit)))
        modes.update(git_modes(vendor_root / path, commit, path + "/"))
    originals = {name: source_bytes(repo, f"config/agent/code/{name}.yaml") for name in ARMS}
    mix = source_bytes(repo, "config/agent/code/mix-four-whitebox.yaml")
    output.mkdir(parents=True, exist_ok=False)
    harness = output / "harness"
    harness.mkdir()
    source_hashes = {}
    symlinks = {}
    component_counts = {}
    with (output / "source.tar.gz").open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as destination:
                for prefix, data in archives:
                    component_counts[prefix.rstrip("/") or "official_root"] = {"files": 0, "symlinks": 0}
                    count = component_counts[prefix.rstrip("/") or "official_root"]
                    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                        for entry in archive:
                            stream = archive.extractfile(entry) if entry.isfile() else None
                            entry.name = prefix + entry.name
                            if stream is not None:
                                contents = stream.read()
                                source_hashes[entry.name] = sha(contents)
                                count["files"] += 1
                                stream = io.BytesIO(contents)
                            elif entry.issym():
                                symlinks[entry.name] = entry.linkname
                                count["symlinks"] += 1
                            destination.addfile(entry, stream)
    if set(modes) != set(source_hashes) | set(symlinks):
        raise ValueError("source manifest does not cover all official and vendor git blobs")
    changes = {}
    for name, original in originals.items():
        generated = modal_profile(original, name)
        with (harness / f"{name}.yaml").open("xb") as file:
            file.write(generated)
        changes[name] = {
            "official_sha256": sha(original),
            "prepared_sha256": sha(generated),
            "diff": "".join(
                difflib.unified_diff(
                    original.decode().splitlines(True),
                    generated.decode().splitlines(True),
                    fromfile=f"official/{name}.yaml",
                    tofile=f"modal/{name}.yaml",
                )
            ),
        }
    for path, data in ((harness / "mix-four-whitebox.yaml", mix), (output / "launch.sh", launch_bytes())):
        with path.open("xb") as file:
            file.write(data)
    manifest = {
        "source_repository": "https://github.com/XiaomiMiMo/verl",
        "source_commit": COMMIT,
        "submodules": PINS,
        "source_archive_sha256": sha((output / "source.tar.gz").read_bytes()),
        "source_file_count": len(source_hashes),
        "source_files_sha256": source_hashes,
        "source_git_modes": modes,
        "source_symlinks": symlinks,
        "source_symlink_count": len(symlinks),
        "source_component_counts": component_counts,
        "source_directory": SOURCE,
        "run_directory": RUN,
        "checkpoint_directory": CHECKPOINT,
        "deadline_utc": DEADLINE,
        "harnesses": changes,
        "mix_sha256": sha(mix),
        "launch_sha256": sha(launch_bytes()),
        "cpu_config_preflight": "pending",
        "training_updates": 0,
        "phases": {
            "fresh": {"total_steps": 1, "wandb_id": "a9off001", "resume_mode": "disable"},
            "resume": {
                "total_steps": 2,
                "wandb_id": "a9off002",
                "resume_mode": "resume_path",
                "resume_from_path": CHECKPOINT + "/global_step_1",
            },
        },
        "limits": [
            "Two-step native Code smoke, not DSH16/64 acceptance or published recipe reproduction.",
            "Fresh total_steps=1 intentionally stops at checkpoint1; "
            "native last-step validation also runs despite test_freq=2.",
            "N4/batch1 and two updates do not guarantee all four step-hash arms execute.",
            "64K cumulative trajectory budget and SGLang context; 32768 tokens per model request.",
        ],
        "external_gates": [
            "source/data/credential identity",
            "real CPU resolved config",
            "GPU model initialization",
            "live deadline guard and approved PIDs",
            "no competing training process",
            "complete step1 model/optimizer/extra/data/TransferQueue checkpoint before resume",
        ],
    }
    with (output / "manifest.json").open("x") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--vendor-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.repository, args.vendor_root, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "source_commit": COMMIT,
                "source_file_count": manifest["source_file_count"],
                "source_archive_sha256": manifest["source_archive_sha256"],
                "cpu_config_preflight": "pending",
            }
        )
    )


if __name__ == "__main__":
    main()
