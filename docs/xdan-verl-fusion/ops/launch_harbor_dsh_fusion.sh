#!/usr/bin/env bash
# Fusion task x harness smoke: Harbor tasks x mixed harness (2 DSH + 2 mini-mimocode per group,
# advantage grouped by harness). DSH is injected at run time (no DSH task images). TB2.1 smoke subset is
# an eval benchmark: pipeline validation only, checkpoints are discarded.
# Prerequisites: start_ray_fusion.sh and a running DSH gateway (public origin file in $DSH_SERVICES_DIR).
# usage: launch_harbor_dsh_fusion.sh fresh|preflight
set -euo pipefail
PREFIX=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
SCRIPTS=/workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/scripts
source "$PREFIX/activate.sh" "$PREFIX"
source "$SCRIPTS/runtime.env"
source /workspace/xdan-verl-fusion/ops/fusion-runtime.env
# Needs the M1b ports (P0 verl fixes + DSH runtime) and the Harbor env; newer than the Music smoke source.
export SOURCE=/workspace/xdan-verl-fusion/source-4faed1ec
export PYTHONPATH="$SOURCE:$SOURCE/third_party/mimoagent-osr/src:$SOURCE/third_party/uni_agent"

export RUN_ID=fusion-harbor-dsh-mix-4gpu-20261002-hd1
export BASE_RUN=$XDAN_FUSION_ROOT/runs/$RUN_ID RUN_DIR=$XDAN_FUSION_ROOT/runs/$RUN_ID
export CHECKPOINT_DIR=$XDAN_FUSION_ROOT/checkpoints/$RUN_ID
export TRAIN_DATA=$XDAN_FUSION_ROOT/data/harbor-tb21-smoke4/train.parquet
export VAL_DATA=$XDAN_FUSION_ROOT/data/harbor-tb21-smoke4/train.parquet
# Reuse the gateway/tunnel started for the DSH Code run.
DSH_SERVICES_DIR=${DSH_SERVICES_DIR:-$XDAN_FUSION_ROOT/runs/fusion-code-dsh-mix-4gpu-20261002-d1}

# DSH gateway: public routing metadata + private route dir (train-dsh-minimal.sh forwards these to Ray).
export DSH_GATEWAY_ROUTE_DIR=/root/mimo-private/dsh-routes
if [[ "${1:-fresh}" == preflight ]]; then
  # CPU only: resolve config and run the recipe validator; no GPU workers, no training.
  export PREFLIGHT_ONLY=1 SKIP_CLUSTER_CHECK=1 CUDA_VISIBLE_DEVICES=""
  export DSH_GATEWAY_PUBLIC_ORIGIN=https://preflight.invalid
  export RUN_DIR=$RUN_DIR/preflight BASE_RUN=$RUN_DIR/preflight
else
  export DSH_GATEWAY_PUBLIC_ORIGIN="$(cat "$DSH_SERVICES_DIR/public-origin.txt")"
fi
test -n "$DSH_GATEWAY_PUBLIC_ORIGIN"

# Task x harness: two Harbor tasks per step, 4 trajectories each = 2 DSH + 2 mini-mimocode.
export MIMOAGENT_HARNESS_SPEC=$SOURCE/config/agent/harbor/mix-dsh-mimocode.yaml
export MIXED_HARNESS_MODE=paired-subgroup MIXED_HARNESS_SEED=20261002
export ALGORITHM_GROUP_ADVANTAGE_BY_HARNESS=True

export TRAINER_MODE=colocate_async
export TRAIN_NNODES=1 TRAIN_NGPUS_PER_NODE=4 ROLLOUT_NNODES=0 ROLLOUT_NGPUS_PER_NODE=4
export ACTOR_TP=4 ACTOR_PP=1 ACTOR_CP=1 ACTOR_EP=1 ROLLOUT_TP=2
export MAXLEN=65536 PROMPT_LENGTH=4096 RESPONSE_LENGTH=61440 PPO_MAX_TOKEN_LEN_PER_GPU=65536
export N=4 TRAIN_BATCH_SIZE=2 PPO_MINI_BATCH_SIZE=2
export TOTAL_STEPS=2 TOTAL_EPOCHS=2 SAVE_FREQ=2 MAX_ACTOR_CKPT_TO_KEEP=1 TEST_FREQ=-1
export MAX_CONCURRENT_SESSIONS=4 ROLLOUT_MAX_RUNNING_REQUESTS=4 AGENT_NUM_WORKERS=1
export TRAJECTORY_TIMEOUT=3600 MODEL_REQUEST_TIMEOUT=3600 MODEL_SDK_MAX_RETRIES=0
export TOOL_CALL_ERROR_PENALTY_ENABLE=false DEEP_FAILURE_MASK_ENABLE=false
export WANDB_RUN_ID=fhd101 EXP_NAME="$RUN_ID-fresh" WANDB_DIR="$RUN_DIR"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard" AGENT_DEBUG_DIR="$RUN_DIR/dumps"
export UNI_AGENT_LOG_DIR="$RUN_DIR/trajectories" ROLLOUT_DATA_DIR="$RUN_DIR/rollouts"
export VALIDATION_DATA_DIR="$RUN_DIR/validation" RESOLVED_CONFIG_PATH="$RUN_DIR/resolved_config.yaml"
export TRAINER_LOGGERS='[console,tensorboard,file,wandb]'
unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN WANDB_API_KEY MODAL_TOKEN_ID MODAL_TOKEN_SECRET
test -f "$MODEL_PATH/config.json" && test ! -e "$CHECKPOINT_DIR/global_step_1"
mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR"

FORWARD_ARGS=()
for name in CUDA_HOME LIBRARY_PATH CUDA_LIB_PATH NETRC MODAL_PROFILE WANDB_ENTITY WANDB_RUN_ID WANDB_MODE WANDB_RESUME WANDB_DIR RUN_DIR VERL_FILE_LOGGER_PATH MIMOAGENT_RG_PATH; do
  FORWARD_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.$name=\"${!name}\"")
done
cd "$SOURCE"
exec bash scripts/code/train-dsh-minimal.sh \
  ++actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=65536 \
  actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
  ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=flash \
  actor_rollout_ref.actor.checkpoint.async_save=False \
  "trainer.logger=$TRAINER_LOGGERS" \
  trainer.resume_mode=disable trainer.resume_from_path=null \
  "${FORWARD_ARGS[@]}"
