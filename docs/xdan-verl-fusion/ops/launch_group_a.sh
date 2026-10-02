#!/usr/bin/env bash
# Group A formal run r1: MiMo Code (holdout removed) + Harbor stage1 in one dataset, harness mix
# {MiMo-Code, DSH (runtime payload)} with step-hash (one harness per group), DAPO filtering on.
# Plan: docs/xdan-verl-fusion/formal-training-plan.md (approved 2026-10-02). The first 5 steps double as
# the measured trial (step time, Modal cost, DSH injection, gateway timeouts).
# Prerequisites: start_ray_fusion.sh, DSH gateway (runs/dsh-gateway-runpod).
# usage: launch_group_a.sh fresh | preflight | resume (RESUME_STEP=n)
set -euo pipefail
PREFIX=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
SCRIPTS=/workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/scripts
source "$PREFIX/activate.sh" "$PREFIX"
source "$SCRIPTS/runtime.env"
source /workspace/xdan-verl-fusion/ops/fusion-runtime.env

PHASE="${1:-fresh}"
export RUN_ID=group-a-r1
export BASE_RUN=$XDAN_FUSION_ROOT/runs/$RUN_ID
export CHECKPOINT_DIR=$XDAN_FUSION_ROOT/checkpoints/$RUN_ID
export TRAIN_DATA=$XDAN_FUSION_ROOT/data/group-a/train-r1.parquet
export VAL_DATA=$XDAN_FUSION_ROOT/data/harbor-stage1/validation.parquet

export DSH_GATEWAY_ROUTE_DIR=/root/mimo-private/dsh-routes
export DSH_GATEWAY_PUBLIC_ORIGIN="$(cat "$XDAN_FUSION_ROOT/runs/dsh-gateway-runpod/public-origin.txt")"
export MIMOAGENT_HARNESS_SPEC=$SOURCE/config/agent/mixed/mix-dsh-mimocode.yaml
export MIXED_HARNESS_MODE=step-hash MIXED_HARNESS_SEED=20261002 ALGORITHM_GROUP_ADVANTAGE_BY_HARNESS=False

export TRAINER_MODE=colocate_async
export TRAIN_NNODES=1 TRAIN_NGPUS_PER_NODE=4 ROLLOUT_NNODES=0 ROLLOUT_NGPUS_PER_NODE=4
export ACTOR_TP=4 ACTOR_PP=1 ACTOR_CP=1 ACTOR_EP=1 ROLLOUT_TP=2
export MAXLEN=65536 PROMPT_LENGTH=4096 RESPONSE_LENGTH=61440 PPO_MAX_TOKEN_LEN_PER_GPU=65536
export N=8 TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 MICRO_BSZ_PER_GPU=1 USE_DYNAMIC_BSZ=True
export MEGATRON_OFFLOAD=True
export LOSS_AGG_MODE=prompt-mean NORM_ADV_BY_STD_IN_GRPO=False ENTROPY_COEFF=0 ENTROPY_CHUNKING=True ENTROPY_CHUNK_SIZE=16384
export FILTER_GROUPS_ENABLE=True FILTER_GROUPS_METRIC=reward
export MAX_OFF_POLICY_THRESHOLD=2 MAX_OFF_POLICY_STRATEGY=drop
export REPETITION_DETECT_ENABLE=true REPETITION_ZERO_REWARD=false REPETITION_PENALTY_ENABLE=false
export TOOL_CALL_ERROR_PENALTY_ENABLE=false DEEP_FAILURE_MASK_ENABLE=false
export TOTAL_STEPS="${TOTAL_STEPS:-100}" TOTAL_EPOCHS=1 SAVE_FREQ=10 MAX_ACTOR_CKPT_TO_KEEP=2 TEST_FREQ=-1
export VAL_N=1 VAL_BATCH_SIZE=8 VAL_DO_SAMPLE=False
export MAX_CONCURRENT_SESSIONS=48 ROLLOUT_MAX_RUNNING_REQUESTS=48 AGENT_NUM_WORKERS=8 GATEWAY_COUNT=1
export TRAJECTORY_TIMEOUT=5400 MODEL_REQUEST_TIMEOUT=3600 MODEL_SDK_MAX_RETRIES=0 HARNESS_TURN_MAX_TOKENS=32768
export TRAINER_LOGGERS='[console,tensorboard,file,wandb]'
unset NVTE_FUSED_ATTN NVTE_FLASH_ATTN NVTE_UNFUSED_ATTN WANDB_API_KEY MODAL_TOKEN_ID MODAL_TOKEN_SECRET

RESUME_ARGS=(trainer.resume_mode=disable trainer.resume_from_path=null)
case "$PHASE" in
  fresh) export WANDB_RUN_ID=ga101 RUN_DIR="$BASE_RUN"
         test ! -e "$CHECKPOINT_DIR/global_step_1" ;;
  preflight) export WANDB_RUN_ID=ga101p RUN_DIR="$BASE_RUN/preflight" BASE_RUN="$BASE_RUN/preflight"
         export PREFLIGHT_ONLY=1 SKIP_CLUSTER_CHECK=1 CUDA_VISIBLE_DEVICES="" ;;
  resume) : "${RESUME_STEP:?}"
         export WANDB_RUN_ID="ga101r${RESUME_STEP}" RUN_DIR="$BASE_RUN/resume-step${RESUME_STEP}"
         test -d "$CHECKPOINT_DIR/global_step_${RESUME_STEP}/actor"
         RESUME_ARGS=(trainer.resume_mode=resume_path "trainer.resume_from_path=$CHECKPOINT_DIR/global_step_${RESUME_STEP}") ;;
  *) echo "usage: launch_group_a.sh fresh|preflight|resume" >&2; exit 2 ;;
esac
export EXP_NAME="$RUN_ID-$PHASE" WANDB_DIR="$RUN_DIR"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard" AGENT_DEBUG_DIR="$RUN_DIR/dumps"
export UNI_AGENT_LOG_DIR="$RUN_DIR/trajectories" ROLLOUT_DATA_DIR="$RUN_DIR/rollouts"
export VALIDATION_DATA_DIR="$RUN_DIR/validation" RESOLVED_CONFIG_PATH="$RUN_DIR/resolved_config.yaml"
test -f "$MODEL_PATH/config.json" && test -f "$TRAIN_DATA"
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
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.checkpoint.async_save=False \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,hf_model,optimizer,extra]' \
  "trainer.logger=$TRAINER_LOGGERS" \
  trainer.val_before_train=False \
  "${FORWARD_ARGS[@]}" "${RESUME_ARGS[@]}"
