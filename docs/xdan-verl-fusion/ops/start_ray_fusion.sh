#!/usr/bin/env bash
# Restart a dedicated Ray head with the fusion PYTHONPATH (workers inherit the head's environment).
set -euo pipefail
PREFIX=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
SCRIPTS=/workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/scripts
source "$PREFIX/activate.sh" "$PREFIX"
source "$SCRIPTS/runtime.env"
source /workspace/xdan-verl-fusion/ops/fusion-runtime.env
ray stop --force >/dev/null 2>&1 || true
sleep 3
ray start --head --port=6381 --num-gpus="$(nvidia-smi -L | wc -l)" \
  --object-store-memory=2147483648 --temp-dir="/tmp/ray-fusion-${1:-$(date -u +%Y%m%d%H%M)}" \
  --dashboard-port=8266 --disable-usage-stats
for _ in 1 2 3 4 5 6; do ray status 2>/dev/null | grep -E "GPU|CPU" && break; sleep 5; done || true
