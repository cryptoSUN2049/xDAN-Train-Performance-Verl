#!/usr/bin/env bash
# Two-step integration preset; GPU memory/runtime compatibility still needs a real probe.
set -euo pipefail

: "${DSH_GATEWAY_PUBLIC_ORIGIN:?DSH_GATEWAY_PUBLIC_ORIGIN must identify the public session proxy}"
: "${DSH_GATEWAY_ROUTE_DIR:?DSH_GATEWAY_ROUTE_DIR must identify the shared proxy route directory}"

# Two GPUs use GPU optimizer steps by default; CPU optimization is opt-in.
case "${CPU_OPTIMIZER_OFFLOAD-False}" in
  [Ff][Aa][Ll][Ss][Ee])
    export CPU_OPTIMIZER_OFFLOAD=False
    OPTIMIZER_OFFLOAD_FRACTION=0.0
    ;;
  [Tt][Rr][Uu][Ee])
    export CPU_OPTIMIZER_OFFLOAD=True
    OPTIMIZER_OFFLOAD_FRACTION=1.0
    ;;
  *)
    echo "CPU_OPTIMIZER_OFFLOAD must be True or False" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MIMOAGENT_HARNESS_SPEC="${MIMOAGENT_HARNESS_SPEC:-${REPO_ROOT}/config/agent/code/dsh-only.yaml}"
export EXP_NAME="${EXP_NAME:-dsh-minimal}"

export TRAIN_NNODES="${TRAIN_NNODES:-1}"
export TRAIN_NGPUS_PER_NODE="${TRAIN_NGPUS_PER_NODE:-2}"
export ROLLOUT_NNODES="${ROLLOUT_NNODES:-0}"
export ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-${TRAIN_NGPUS_PER_NODE}}"
export ACTOR_TP="${ACTOR_TP:-${TRAIN_NGPUS_PER_NODE}}"
export ACTOR_PP="${ACTOR_PP:-1}"
export ACTOR_CP="${ACTOR_CP:-1}"
export ACTOR_EP="${ACTOR_EP:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-${TRAIN_NGPUS_PER_NODE}}"
export MEGATRON_OFFLOAD="${MEGATRON_OFFLOAD:-True}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.60}"

export N="${N:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
export MICRO_BSZ_PER_GPU="${MICRO_BSZ_PER_GPU:-1}"
export TOTAL_STEPS="${TOTAL_STEPS:-2}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
export MAXLEN="${MAXLEN:-65536}"
export PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
export HARNESS_TURN_MAX_TOKENS="${HARNESS_TURN_MAX_TOKENS:-4096}"
export ENTROPY_CHUNK_SIZE="${ENTROPY_CHUNK_SIZE:-4096}"
export FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-False}"
export REPETITION_DETECT_ENABLE="${REPETITION_DETECT_ENABLE:-false}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
export VAL_N="${VAL_N:-1}"
export SAVE_FREQ="${SAVE_FREQ:-1}"
export TEST_FREQ="${TEST_FREQ:--1}"

export AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-1}"
export GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
export MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-1}"
export ROLLOUT_MAX_RUNNING_REQUESTS="${ROLLOUT_MAX_RUNNING_REQUESTS:-1}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-4096}"
export SGLANG_MAX_PREFILL_TOKENS="${SGLANG_MAX_PREFILL_TOKENS:-4096}"
export MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-16}"
export TRAJECTORY_TIMEOUT="${TRAJECTORY_TIMEOUT:-3600}"

# MEGATRON_OFFLOAD stages states between rollout/training; it does not select
# the optimizer compute device. Both choices retain FP32 gradients/moments and
# full master precision; GPU TE keeps its default lossless BF16 + int16 remainder.
# The Bridge provider DDP default is BF16, so explicitly request FP32 reduction.
OVERRIDES=(
  "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${CPU_OPTIMIZER_OFFLOAD}"
  "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION}"
  actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True
  ++actor_rollout_ref.actor.megatron.override_ddp_config.grad_reduce_in_fp32=True
  actor_rollout_ref.actor.optim.main_grads_dtype=fp32
  actor_rollout_ref.actor.optim.exp_avg_dtype=fp32
  actor_rollout_ref.actor.optim.exp_avg_sq_dtype=fp32
)
# Only public routing metadata and a credential FILE PATH enter Hydra/Ray config.
# Keep Modal tokens in the credential file accessible to Ray workers.
for name in DSH_GATEWAY_PUBLIC_ORIGIN DSH_GATEWAY_ROUTE_DIR MODAL_CONFIG_PATH; do
  value="${!name:-}"
  if [[ -n "${value}" ]]; then
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    OVERRIDES+=("+ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${value}\"")
  fi
done

# Existing launcher preserves cluster checks, resolved config, manifests and
# checkpoint paths. Explicit trailing Hydra overrides also permit resume_path.
exec bash "${REPO_ROOT}/scripts/code/train.sh" "${OVERRIDES[@]}" "$@"
