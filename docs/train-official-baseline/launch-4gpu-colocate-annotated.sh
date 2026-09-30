#!/usr/bin/env bash
set -euo pipefail
# Official source a2ad9f6160b03ff2d47e59832bfb6b289f37c917; immutable deadline 2026-09-30T12:51:43.743Z.
# 历史 2026-09-30 配置：恢复服务器前重新绑定 run/W&B/deadline/守护，不直接执行旧 run。
# 三种 V1 模式见 script-comparison.md；当前走官方 colocate_async 基线。
# uv: reuse the verified project stack; a new Pod must restore it first.
# /opt is ephemeral; persistent archives/wheels live in /workspace/env_infra/rtx6000/.
export UV_ENV_DIR=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/venv
# 官方源码 a2ad9f61，核心/两个子模块保持原字节；本脚本仅设置运行配置。
export SOURCE=/workspace/train-p0-dsh-integration/source-a2ad9f61
export BASE_RUN=/workspace/train-p0-dsh-integration/runs/official-code-4gpu-64k-r1-20260930
export RUN_ID=official-code-4gpu-64k-r1-20260930
# checkpoint 在 /opt 活跃写入；关 Pod 前必须校验并备份到 /workspace。
export CHECKPOINT_DIR=/opt/train-p0-dsh-integration/checkpoints/official-code-4gpu-64k-r1-20260930
export MODEL_PATH=/workspace/models/MiMo-V2.6-Distill-Qwen-9B
export TRAIN_DATA=/workspace/train-p0-dsh-integration/data-train8-is1-r1/train.parquet
export VAL_DATA=/workspace/train-p0-dsh-integration/data-minimal-r1/holdout.parquet
# 认证仅传私有文件路径；W&B/Modal token 不进入 Hydra argv 或仓库。
export NETRC=/root/mimo-private/wandb.netrc
export MODAL_CONFIG_PATH=/root/mimo-private/modal.toml
export MODAL_PROFILE=l98348740
# 原生 W&B + RL-Insight logger；服务可达还需真实 step/trace 验收。
export RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080
export VERL_RL_INSIGHT_ENABLE=1
export MIMOAGENT_RG_PATH=/workspace/train-p0-dsh-integration/tools/mimoagent-cache/ripgrep/15.1.0/rg
export RAY_INIT_ADDRESS=127.0.0.1:6381
export PROJECT_NAME=xDAN-Train-Performance-Verl
export WANDB_ENTITY=xdan-ai
export WANDB_MODE=online
export WANDB_RESUME=never
# 官方 Code validator 要求 colocate_async；4 张卡在采样/训练阶段共享。
# ROLLOUT_NNODES=0 表示没有独立采样池，不表示不采样，也不是额外再租 4 张卡。
# DSH 扩展脚本为 separate_async，独立训练 2 卡 + 采样 2 卡。
export TRAINER_MODE=colocate_async
export TRAIN_NNODES=1
export TRAIN_NGPUS_PER_NODE=4
export ROLLOUT_NNODES=0
export ROLLOUT_NGPUS_PER_NODE=4
# 训练 TP4/CP1；采样 TP2，四卡采样阶段可形成 2 个 TP2 replica。
export ACTOR_TP=4
export ACTOR_PP=1
export ACTOR_CP=1
export ACTOR_EP=1
export ROLLOUT_TP=2
# 64K 是训练模型上下文/轨迹预算：4096 prompt + 61440 response。
# SGLang context_length 也须显式设置 65536；单次模型请求上限另设 32768。
export MAXLEN=65536
export PROMPT_LENGTH=4096
export RESPONSE_LENGTH=61440
export PPO_MAX_TOKEN_LEN_PER_GPU=65536
# 最小闭环容量配置 N4/batch1/mini1/micro1/static；官方默认 N16/batch64。
# reward 全相同的组可被原版 filter_groups 丢弃；配置可用不等于已有有效更新。
export N=4
export TRAIN_BATCH_SIZE=1
export PPO_MINI_BATCH_SIZE=1
export MICRO_BSZ_PER_GPU=1
export USE_DYNAMIC_BSZ=False
# 沿用官方阶段 offload；它不能代替检查 optimizer 的 CPU/GPU step 设置。
export MEGATRON_OFFLOAD=True
# 每个真实 step 保存；fresh=1，resume=2，从本轮 CP1 恢复。
# 原版最后一步仍触发 validation；两步闭环不保证四种 harness 都已覆盖。
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
# 官方四原生 harness: mini-mimocode / mini-bash / mini-claude-code / mini-codex。
# step-hash: 同一 prompt 的 N 条轨迹选同一个 arm；保留原版 GRPO 设置。
# DSH paired-subgroup: 同一组 DSH2 + MiMo2，并按 harness 分组算 advantage。
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
# 单 turn 32768，沿用官方；DSH SDK 的单 turn 默认 4096，二者都不等于总上下文。
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
# fresh 禁用旧 checkpoint；resume 只加载本次完整 CP1，必须取得恢复后的真实 step2。
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
  # CPU 只解析配置/原版 validator；不创建 GPU worker 或训练，也不证明模型容量。
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
# 已有 Ray head 不会自动继承当前 shell：显式转发认证路径、run 身份、监控地址。
for name in NETRC MODAL_CONFIG_PATH MODAL_PROFILE WANDB_ENTITY WANDB_RUN_ID WANDB_MODE WANDB_RESUME WANDB_DIR RUN_DIR VERL_FILE_LOGGER_PATH RL_INSIGHT_SERVER_URL VERL_RL_INSIGHT_ENABLE MIMOAGENT_RG_PATH; do
  FORWARD_ARGS+=("+ray_kwargs.ray_init.runtime_env.env_vars.$name=\"${!name}\"")
done
# 原版入口 -> recipes/code/run_train.sh -> verl.trainer.main_ppo；不用 DSH runner/custom sampler。
exec bash scripts/code/train.sh \
  ++actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=65536 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.actor.checkpoint.async_save=False \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra]' \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model,hf_model,optimizer,extra]' \
  'trainer.logger=[console,tensorboard,file,wandb,rl_insight]' \
  "${FORWARD_ARGS[@]}" "${RESUME_ARGS[@]}"
