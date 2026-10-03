#!/usr/bin/env bash
# Preserve one checkpoint of a running run as the init point of a new data phase:
# hard-links (falls back to copy) global_step_<N>/actor into checkpoints/<dst>/global_step_<N>/actor once step N
# is fully committed. data.pt and transfer_queue/ are deliberately NOT carried over: the new phase uses another
# dataset, and the async trainer may only reset its dataloader when no old in-flight prompts are restored.
# Append-only; never deletes. usage: preserve_resume_ckpt.sh <src_run> <dst_run> <step>
set -uo pipefail
SRC=/workspace/xdan-verl-fusion/checkpoints/$1
DST=/workspace/xdan-verl-fusion/checkpoints/$2/global_step_$3
N=$3
log() { echo "[$(date -u +%FT%TZ)] $*"; }
log "waiting for $SRC/global_step_$N"
while :; do
  latest=$(cat "$SRC/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  if [ "$latest" -ge "$N" ] && [ -f "$SRC/global_step_$N/actor/ckpt_contents.json" ]; then break; fi
  if [ "$latest" -gt "$N" ] && [ ! -d "$SRC/global_step_$N/actor" ]; then log "step $N rotated away before it could be kept"; exit 1; fi
  sleep 60
done
mkdir -p "$DST"
if cp -al "$SRC/global_step_$N/actor" "$DST/actor" 2>/dev/null; then how=hardlink; else rm -rf "$DST/actor"; cp -a "$SRC/global_step_$N/actor" "$DST/actor"; how=copy; fi
echo "$N" > "$DST/../latest_checkpointed_iteration.txt"
log "kept step $N actor via $how: $(du -sh "$DST/actor" | cut -f1); files src=$(find "$SRC/global_step_$N/actor" -type f | wc -l) dst=$(find "$DST/actor" -type f | wc -l)"
