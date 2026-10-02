#!/usr/bin/env bash
# Group A evaluation (val_only, no training): TB2.1 (89) + MiMo Code holdout100 under one harness,
# n samples per task with training-time sampling. Used for the SFT baseline and every RL checkpoint.
# Prerequisites: start_ray_fusion.sh; for HARNESS=dsh, the DSH gateway (runs/dsh-gateway-runpod).
# usage: HARNESS=mimocode|dsh [MODEL=<hf dir>] [TAG=sft] [VAL_N=4] launch_eval_a.sh
set -euo pipefail
PREFIX=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
SCRIPTS=/workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/scripts
source "$PREFIX/activate.sh" "$PREFIX"
source "$SCRIPTS/runtime.env"
source /workspace/xdan-verl-fusion/ops/fusion-runtime.env

HARNESS="${HARNESS:?mimocode or dsh}"
TAG="${TAG:-sft}"
export MODEL_PATH="${MODEL:-$MODEL_PATH}"
export RUN_ID="eval-a-$TAG-$HARNESS"
export RUN_DIR=$XDAN_FUSION_ROOT/runs/$RUN_ID BASE_RUN=$XDAN_FUSION_ROOT/runs/$RUN_ID
export CHECKPOINT_DIR=$XDAN_FUSION_ROOT/checkpoints/$RUN_ID
export VAL_DATA=$XDAN_FUSION_ROOT/data/eval-a/tb21+code-holdout100.parquet
export TRAIN_DATA=$VAL_DATA
case "$HARNESS" in
  mimocode) export MIMOAGENT_HARNESS_SPEC=$SOURCE/config/agent/mixed/mimocode-only.yaml ;;
  dsh) export MIMOAGENT_HARNESS_SPEC=$SOURCE/config/agent/mixed/dsh-only.yaml ;;
  *) echo "HARNESS must be mimocode or dsh" >&2; exit 2 ;;
esac
export DSH_GATEWAY_ROUTE_DIR=/root/mimo-private/dsh-routes
export DSH_GATEWAY_PUBLIC_ORIGIN="$(cat "$XDAN_FUSION_ROOT/runs/dsh-gateway-runpod/public-origin.txt")"
export MIXED_HARNESS_MODE=step-hash MIXED_HARNESS_SEED=20261002 ALGORITHM_GROUP_ADVANTAGE_BY_HARNESS=False

export TRAINER_MODE=colocate_async
export TRAIN_NNODES=1 TRAIN_NGPUS_PER_NODE=4 ROLLOUT_NNODES=0 ROLLOUT_NGPUS_PER_NODE=4
export ACTOR_TP=4 ACTOR_PP=1 ACTOR_CP=1 ACTOR_EP=1 ROLLOUT_TP=2
export MAXLEN=65536 PROMPT_LENGTH=4096 RESPONSE_LENGTH=61440 PPO_MAX_TOKEN_LEN_PER_GPU=65536
export N=4 TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4
export VAL_N="${VAL_N:-4}" VAL_DO_SAMPLE=True VAL_BATCH_SIZE=189
export TOTAL_STEPS=1 TOTAL_EPOCHS=1 SAVE_FREQ=-1 TEST_FREQ=-1
export MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-48}" ROLLOUT_MAX_RUNNING_REQUESTS=48 AGENT_NUM_WORKERS=8
export TRAJECTORY_TIMEOUT=5400 MODEL_REQUEST_TIMEOUT=3600 MODEL_SDK_MAX_RETRIES=0 HARNESS_TURN_MAX_TOKENS=32768
export WANDB_RUN_ID="ea-$TAG-$HARNESS" EXP_NAME="$RUN_ID" WANDB_DIR="$RUN_DIR"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard" AGENT_DEBUG_DIR="$RUN_DIR/dumps"
export UNI_AGENT_LOG_DIR="$RUN_DIR/trajectories" ROLLOUT_DATA_DIR="$RUN_DIR/rollouts"
export VALIDATION_DATA_DIR="$RUN_DIR/validation" RESOLVED_CONFIG_PATH="$RUN_DIR/resolved_config.yaml"
export TRAINER_LOGGERS='[console,tensorboard,file,wandb]'
# Own Modal app so this line's sandboxes can be listed and cleaned without touching other runs.
export MODAL_APP_NAME=xdan-fusion-eval-a
unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN WANDB_API_KEY MODAL_TOKEN_ID MODAL_TOKEN_SECRET
test -f "$MODEL_PATH/config.json"
mkdir -p "$RUN_DIR"

FORWARD_ARGS=()
for name in CUDA_HOME LIBRARY_PATH CUDA_LIB_PATH NETRC MODAL_CONFIG_PATH MODAL_PROFILE MODAL_APP_NAME DSH_GATEWAY_PUBLIC_ORIGIN DSH_GATEWAY_ROUTE_DIR WANDB_ENTITY WANDB_RUN_ID WANDB_MODE WANDB_RESUME WANDB_DIR RUN_DIR VERL_FILE_LOGGER_PATH MIMOAGENT_RG_PATH; do
  FORWARD_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.$name=\"${!name}\"")
done
cd "$SOURCE"
# Official Code entry (the accepted r2 path). Not train-dsh-minimal.sh: its 2-GPU memory defaults
# (prefill 4096, mamba cache 16, rollout mem 0.6, optimizer overrides) throttled attempt 1 to 23 tok/s.
# DSH needs only the gateway/Modal variables, forwarded above.
exec bash scripts/code/train.sh \
  ++actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=65536 \
  actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
  ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=flash \
  "trainer.logger=$TRAINER_LOGGERS" \
  trainer.resume_mode=disable trainer.resume_from_path=null \
  trainer.val_only=True trainer.val_before_train=True \
  "${FORWARD_ARGS[@]}"
