#!/usr/bin/env bash
# Serial GPU queue for the fusion line on this pod:
#   Music resume (already running) -> Harbor x mimocode -> Code x {DSH + mimocode}.
# Each stage writes exit-code.txt in its run dir; a failed stage does not stop the next one.
set -uo pipefail
OPS=/workspace/xdan-verl-fusion/ops
RUNS=/workspace/xdan-verl-fusion/runs
log() { echo "$(date -u +%FT%TZ) $*"; }

run_stage() {  # name run_dir command...
  local name="$1" dir="$2"; shift 2
  mkdir -p "$dir"
  date -u +%FT%TZ > "$dir/started-utc.txt"
  log "start $name"
  "$@" > "$dir/training.log" 2>&1
  local rc=$?
  echo "$rc" > "$dir/exit-code.txt"; date -u +%FT%TZ > "$dir/ended-utc.txt"
  log "end $name rc=$rc"
}

MUSIC=$RUNS/fusion-music-4gpu-20261002-s1/resume-step1
while [ ! -f "$MUSIC/exit-code.txt" ]; do sleep 30; done
log "music resume finished rc=$(cat "$MUSIC/exit-code.txt")"

bash "$OPS/start_ray_fusion.sh" h1 > "$RUNS/ray-h1.log" 2>&1
run_stage harbor "$RUNS/fusion-harbor-mimocode-4gpu-20261002-h1" bash "$OPS/launch_harbor_fusion.sh" fresh

D1=$RUNS/fusion-code-dsh-mix-4gpu-20261002-d1
bash "$OPS/start_ray_fusion.sh" d1 > "$RUNS/ray-d1.log" 2>&1
if bash "$OPS/start_dsh_services.sh" "$D1" > "$D1/services.log" 2>&1; then
  run_stage dsh "$D1" bash "$OPS/launch_dsh_fusion.sh" fresh
else
  log "dsh services failed; see $D1/services.log"
fi
log "queue done"
