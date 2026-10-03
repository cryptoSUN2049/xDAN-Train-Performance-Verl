#!/usr/bin/env bash
# Keep the HF weights of eval-point checkpoints that checkpoint rotation (MAX_ACTOR_CKPT_TO_KEEP=2,
# SAVE_FREQ=5) would delete: copies global_step_<N>/actor/model/huggingface (~18 GB) to
# checkpoints/<run>-milestones/step_<N> once step N is fully committed. Append-only; never deletes.
# usage (on the pod): setsid nohup bash milestone_keeper.sh [run] [steps...] > keeper.log 2>&1 &
set -uo pipefail
RUN="${1:-group-a-r1}"
shift || true
STEPS=("$@")
[ ${#STEPS[@]} -gt 0 ] || STEPS=(25 50 100)
C=/workspace/xdan-verl-fusion/checkpoints/$RUN
M=/workspace/xdan-verl-fusion/checkpoints/$RUN-milestones
mkdir -p "$M"
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log "keeping steps ${STEPS[*]} of $C into $M"
while :; do
  pending=0
  latest=$(cat "$C/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  for n in "${STEPS[@]}"; do
    [ -f "$M/step_$n/config.json" ] || [ -f "$M/step_$n.MISSED" ] && continue
    pending=1
    src=$C/global_step_$n/actor/model/huggingface
    # latest >= n means the save of step n finished (the tracker is written after the save)
    if [ "$latest" -ge "$n" ] && [ -f "$src/config.json" ]; then
      rm -rf "$M/.step_$n.tmp"
      if cp -a "$src" "$M/.step_$n.tmp" && mv "$M/.step_$n.tmp" "$M/step_$n"; then
        log "kept step $n ($(du -sh "$M/step_$n" | cut -f1))"
      else
        log "copy of step $n failed; will retry"
      fi
    elif [ "$latest" -gt "$n" ] && [ ! -d "$C/global_step_$n" ]; then
      log "step $n already rotated away before it could be kept"
      touch "$M/step_$n.MISSED"
    fi
  done
  [ "$pending" = 0 ] && { log "all milestones kept"; exit 0; }
  sleep 300
done
