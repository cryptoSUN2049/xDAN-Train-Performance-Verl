#!/usr/bin/env bash
# SFT baseline for group A: MiMo-Code then DSH, each on TB2.1 + Code holdout100 with n=4.
set -uo pipefail
OPS=/workspace/xdan-verl-fusion/ops
RUNS=/workspace/xdan-verl-fusion/runs
for harness in mimocode dsh; do
  R=$RUNS/eval-a-sft-$harness
  mkdir -p "$R"
  date -u +%FT%TZ > "$R/started-utc.txt"
  echo "$(date -u +%FT%TZ) start $harness"
  HARNESS=$harness TAG=sft bash "$OPS/launch_eval_a.sh" > "$R/training.log" 2>&1
  rc=$?
  echo "$rc" > "$R/exit-code.txt"
  date -u +%FT%TZ > "$R/ended-utc.txt"
  echo "$(date -u +%FT%TZ) end $harness rc=$rc"
  bash "$OPS/start_ray_fusion.sh" "eval-$harness-done" > "$RUNS/ray-eval-$harness.log" 2>&1
done
echo "eval queue done"
