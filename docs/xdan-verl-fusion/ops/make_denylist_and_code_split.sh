#!/usr/bin/env bash
# Build the evaluation deny list and the holdout-free Code train set on the fusion pod.
set -euo pipefail
P=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
source "$P/activate.sh" "$P" >/dev/null
S=/workspace/xdan-verl-fusion/source-5ecb8cd3
D=/workspace/xdan-verl-fusion/data
V=/workspace/verl-uni-agent-harbor-opd-rl
T=/workspace/train-p0-dsh-integration
OUT=$D/eval-denylist.txt
{
  echo "# Evaluation/holdout identities that must never enter training rows (generated $(date -u +%FT%TZ))."
  echo "# Terminal-Bench 2.1"
  ls "$V/data/harbor/terminal-bench_terminal-bench-2-1/terminal-bench-2-1"
  echo "# SWE-bench Verified (raw name and as a swe-rebench pool identity: same repo__repo-PR naming)"
  for name in $(ls "$V/data-swe-verified-20260921/harbor/swe-bench_swe-bench-verified/swe-bench-verified"); do
    echo "$name"; echo "swe-rebench-v2-fv/$name"
  done
  echo "# eval-set-v1 reservation (source/task)"
  python - <<EOF
import json
for item in json.load(open("$V/data-eval-set-v1/repo/eval-set-v1/selected.json"))["tasks"]:
    print(f"{item['source']}/{item['task']}")
EOF
  echo "# stage1 validation (shared by all stage1 slices)"
  ls "$V/data-pipe-s1/stage1/tasks-validation"
  echo "# MiMo Code holdout100"
  python - <<EOF
import pandas as pd
for info in pd.read_parquet("$T/data-tiers-20261002/code/holdout100.parquet")["extra_info"]:
    print(info["instance_id"])
EOF
} > "$OUT"
echo "deny list: $(grep -vc '^#' "$OUT") identities -> $OUT"

mkdir -p "$D/code-clean"
cd "$S"
PYTHONPATH="$S" python scripts/data/exclude_holdout.py \
  --train "$T/data-full-code-20261001/train.parquet" \
  --holdout "$T/data-tiers-20261002/code/holdout100.parquet" \
  --out "$D/code-clean/train-minus-holdout100.parquet"
