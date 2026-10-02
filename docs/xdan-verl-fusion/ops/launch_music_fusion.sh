#!/usr/bin/env bash
# Fusion smoke: Music 4-GPU on xdan/fusion-a9f2985 (6c702c5b), same parameters as
# official-music-4gpu-20261002-m1 (a2ad9f61) so step metrics are directly comparable.
# Differences vs m1: fusion source, no RL-Insight logger, no val before train, bounded steps.
# usage: launch_music_fusion.sh fresh | resume (RESUME_STEP=n)
set -euo pipefail
PREFIX=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
SCRIPTS=/workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/scripts
source "$PREFIX/activate.sh" "$PREFIX"
source "$SCRIPTS/runtime.env"
source /workspace/xdan-verl-fusion/ops/fusion-runtime.env
export PATH="$ABC2MIDI_DIR:$PATH"

export RUN_ID=fusion-music-4gpu-20261002-s1
export BASE_RUN=$XDAN_FUSION_ROOT/runs/$RUN_ID
export CHECKPOINT_DIR=$XDAN_FUSION_ROOT/checkpoints/$RUN_ID
export TRAIN_DATA=/workspace/train-p0-dsh-integration/data-music-split-20261002/train.parquet
export VAL_DATA=/workspace/train-p0-dsh-integration/data-music-split-20261002/holdout.parquet

export TRAIN_NNODES=1 TRAIN_NGPUS_PER_NODE=4 ACTOR_TP=4 ACTOR_PP=1 ACTOR_CP=1 ROLLOUT_TP=2
export MAXLEN=116384 PROMPT_LENGTH=16384
export N=8 TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=4
export ROLLOUT_GPU_MEM_UTIL=0.78 ROLLOUT_MAX_RUNNING_REQUESTS=32 AGENT_NUM_WORKERS=8 REWARD_NUM_WORKERS=16
export TOTAL_EPOCHS=4 SAVE_FREQ=1 TEST_FREQ=-1
export PROJECT_NAME=xDAN-Train-Performance-Verl

PHASE="${1:-fresh}"
case "$PHASE" in
  fresh) export WANDB_RUN_ID=fm101 RUN_DIR="$BASE_RUN" TOTAL_STEPS=1
         RESUME_ARGS=(trainer.resume_mode=disable trainer.resume_from_path=null)
         test ! -e "$CHECKPOINT_DIR/global_step_1" ;;
  resume) : "${RESUME_STEP:?}"
         export WANDB_RUN_ID="fm101r${RESUME_STEP}" RUN_DIR="$BASE_RUN/resume-step${RESUME_STEP}"
         export TOTAL_STEPS=$((RESUME_STEP + 1))
         test -d "$CHECKPOINT_DIR/global_step_${RESUME_STEP}/actor"
         RESUME_ARGS=(trainer.resume_mode=resume_path "trainer.resume_from_path=$CHECKPOINT_DIR/global_step_${RESUME_STEP}") ;;
  *) echo "usage: launch_music_fusion.sh fresh|resume" >&2; exit 2 ;;
esac
export EXP_NAME="$RUN_ID-$PHASE" WANDB_DIR="$RUN_DIR" TENSORBOARD_DIR="$RUN_DIR/tensorboard"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
export ROLLOUT_DATA_DIR="$RUN_DIR/rollouts" VALIDATION_DATA_DIR="$RUN_DIR/validation" RESOLVED_CONFIG_PATH="$RUN_DIR/resolved_config.yaml"
mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR"
test -x "$ABC2MIDI_DIR/abc2midi" && test -f "$MODEL_PATH/config.json"

FORWARD_ARGS=()
for name in CUDA_HOME LIBRARY_PATH CUDA_LIB_PATH CUDA_DEVICE_MAX_CONNECTIONS NETRC WANDB_ENTITY WANDB_RUN_ID WANDB_MODE WANDB_RESUME WANDB_DIR RUN_DIR VERL_FILE_LOGGER_PATH; do
  FORWARD_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.$name=\"${!name}\"")
done
cd "$SOURCE"
exec bash scripts/design/music.sh \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.val_before_train=False \
  actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
  ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=flash \
  actor_rollout_ref.actor.checkpoint.async_save=False \
  'trainer.logger=[console,tensorboard,file,wandb]' \
  "${FORWARD_ARGS[@]}" "${RESUME_ARGS[@]}"
