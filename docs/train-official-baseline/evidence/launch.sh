#!/usr/bin/env bash
set -euo pipefail
# Official source a2ad9f6160b03ff2d47e59832bfb6b289f37c917; immutable deadline 2026-09-30T12:51:43.743Z.
export UV_ENV_DIR=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/venv
export SOURCE=/workspace/train-p0-dsh-integration/source-a2ad9f61
export BASE_RUN=/workspace/train-p0-dsh-integration/runs/official-code-4gpu-64k-r1-20260930
export RUN_ID=official-code-4gpu-64k-r1-20260930
export CHECKPOINT_DIR=/opt/train-p0-dsh-integration/checkpoints/official-code-4gpu-64k-r1-20260930
export MODEL_PATH=/workspace/models/MiMo-V2.6-Distill-Qwen-9B
export TRAIN_DATA=/workspace/train-p0-dsh-integration/data-train8-is1-r1/train.parquet
export VAL_DATA=/workspace/train-p0-dsh-integration/data-minimal-r1/holdout.parquet
export NETRC=/root/mimo-private/wandb.netrc
export MODAL_CONFIG_PATH=/root/mimo-private/modal.toml
export MODAL_PROFILE=l98348740
export RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080
export VERL_RL_INSIGHT_ENABLE=1
export MIMOAGENT_RG_PATH=/workspace/train-p0-dsh-integration/tools/mimoagent-cache/ripgrep/15.1.0/rg
export RAY_INIT_ADDRESS=127.0.0.1:6381
export PROJECT_NAME=xDAN-Train-Performance-Verl
export WANDB_ENTITY=xdan-ai
export WANDB_MODE=online
export WANDB_RESUME=never
export TRAINER_MODE=colocate_async
export TRAIN_NNODES=1
export TRAIN_NGPUS_PER_NODE=4
export ROLLOUT_NNODES=0
export ROLLOUT_NGPUS_PER_NODE=4
export ACTOR_TP=4
export ACTOR_PP=1
export ACTOR_CP=1
export ACTOR_EP=1
export ROLLOUT_TP=2
export MAXLEN=65536
export PROMPT_LENGTH=4096
export RESPONSE_LENGTH=61440
export PPO_MAX_TOKEN_LEN_PER_GPU=65536
export N=4
export TRAIN_BATCH_SIZE=1
export PPO_MINI_BATCH_SIZE=1
export MICRO_BSZ_PER_GPU=1
export USE_DYNAMIC_BSZ=False
export MEGATRON_OFFLOAD=True
export SAVE_FREQ=1
export TEST_FREQ=2
export TOTAL_EPOCHS=10
export VAL_N=1
export VAL_BATCH_SIZE=1
export VAL_DO_SAMPLE=False
export AGENT_NUM_WORKERS=1
export GATEWAY_COUNT=1
export MAX_CONCURRENT_SESSIONS=4
export ROLLOUT_MAX_RUNNING_REQUESTS=4
export MIXED_HARNESS_MODE=step-hash
export MIXED_HARNESS_SEED=20260911
export ALGORITHM_GROUP_ADVANTAGE_BY_HARNESS=False
export LOSS_AGG_MODE=prompt-mean
export NORM_ADV_BY_STD_IN_GRPO=False
export FILTER_GROUPS_ENABLE=True
export FILTER_GROUPS_METRIC=reward
export ENTROPY_COEFF=0
export ENTROPY_CHUNKING=True
export ENTROPY_CHUNK_SIZE=16384
export MAX_OFF_POLICY_THRESHOLD=2
export MAX_OFF_POLICY_STRATEGY=drop
export REPETITION_DETECT_ENABLE=true
export REPETITION_ZERO_REWARD=false
export REPETITION_PENALTY_ENABLE=false
export TOOL_CALL_ERROR_PENALTY_ENABLE=false
export DEEP_FAILURE_MASK_ENABLE=false
export HARNESS_TURN_MAX_TOKENS=32768
export MODEL_REQUEST_TIMEOUT=3600
export MODEL_SDK_MAX_RETRIES=0
export PATH="$UV_ENV_DIR/bin:$PATH"
test "$(command -v python3)" = "$UV_ENV_DIR/bin/python3"
export PYTHONPATH="$SOURCE:$SOURCE/third_party/mimoagent-osr/src:$SOURCE/third_party/uni_agent"
export MIMOAGENT_HARNESS_SPEC="$BASE_RUN/harness/mix-four-whitebox.yaml"
unset WANDB_API_KEY MODAL_TOKEN_ID MODAL_TOKEN_SECRET
PHASE="${1:-fresh}"
test "$#" -le 1
RESUME_ARGS=(trainer.resume_mode=disable trainer.resume_from_path=null)
case "$PHASE" in
  fresh) export TOTAL_STEPS=1 WANDB_RUN_ID=a9off001 RUN_DIR="$BASE_RUN" ;;
  resume) export TOTAL_STEPS=2 WANDB_RUN_ID=a9off002 RUN_DIR="$BASE_RUN/resume-step2"
    RESUME_ARGS=(trainer.resume_mode=resume_path "trainer.resume_from_path=$CHECKPOINT_DIR/global_step_1") ;;
  *) echo "usage: launch.sh [fresh|resume]" >&2; exit 2 ;;
esac
export EXP_NAME="$RUN_ID-$PHASE" WANDB_DIR="$RUN_DIR"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard" AGENT_DEBUG_DIR="$RUN_DIR/dumps"
export UNI_AGENT_LOG_DIR="$RUN_DIR/trajectories" ROLLOUT_DATA_DIR="$RUN_DIR/rollouts"
export VALIDATION_DATA_DIR="$RUN_DIR/validation" RESOLVED_CONFIG_PATH="$RUN_DIR/resolved_config.yaml"
if [[ "${CPU_CONFIG_PREFLIGHT:-0}" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES="" PREFLIGHT_ONLY=1 SKIP_CLUSTER_CHECK=1
  export RESOLVED_CONFIG_PATH="$RUN_DIR/cpu-resolved-config.yaml"
else
  test "${SKIP_CLUSTER_CHECK:-0}" != 1
  test -f "$MODEL_PATH/config.json"
  if [[ "$PHASE" == resume ]]; then
    test -d "$CHECKPOINT_DIR/global_step_1/actor"
    test -f "$CHECKPOINT_DIR/global_step_1/data.pt"
  else test ! -e "$CHECKPOINT_DIR/global_step_1"; fi
fi
cd "$SOURCE"
# The root operator must verify deadline guard and full checkpoint receipts before fit.
FORWARD_ARGS=()
for name in NETRC MODAL_CONFIG_PATH MODAL_PROFILE WANDB_ENTITY WANDB_RUN_ID WANDB_MODE WANDB_RESUME WANDB_DIR RUN_DIR VERL_FILE_LOGGER_PATH RL_INSIGHT_SERVER_URL VERL_RL_INSIGHT_ENABLE MIMOAGENT_RG_PATH; do
  FORWARD_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.$name=\"${!name}\"")
done
exec bash scripts/code/train.sh \
  ++actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=65536 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.actor.checkpoint.async_save=False \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,hf_model,optimizer,extra]' \
  'trainer.logger=[console,tensorboard,file,wandb,rl_insight]' \
  "${FORWARD_ARGS[@]}" "${RESUME_ARGS[@]}"
