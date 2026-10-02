#!/usr/bin/env bash
# Second fusion queue: after the first queue finishes, run Harbor x {DSH + mimocode}.
set -uo pipefail
OPS=/workspace/xdan-verl-fusion/ops
RUNS=/workspace/xdan-verl-fusion/runs
log() { echo "$(date -u +%FT%TZ) $*"; }
until grep -q "queue done" "$RUNS/gpu-queue-fusion.log"; do sleep 30; done
D=$RUNS/fusion-harbor-dsh-mix-4gpu-20261002-hd1
mkdir -p "$D"
if [ ! -s "$RUNS/fusion-code-dsh-mix-4gpu-20261002-d1/public-origin.txt" ]; then
  log "no DSH gateway from the first queue; starting one for this run"
  bash "$OPS/start_dsh_services.sh" "$D/services" > "$D/services.log" 2>&1 || { log "dsh services failed"; exit 1; }
  export DSH_SERVICES_DIR="$D/services"
fi
bash "$OPS/start_ray_fusion.sh" hd1 > "$RUNS/ray-hd1.log" 2>&1
date -u +%FT%TZ > "$D/started-utc.txt"
log "start harbor-dsh"
bash "$OPS/launch_harbor_dsh_fusion.sh" fresh > "$D/training.log" 2>&1
rc=$?; echo "$rc" > "$D/exit-code.txt"; date -u +%FT%TZ > "$D/ended-utc.txt"
log "end harbor-dsh rc=$rc"
log "queue2 done"
