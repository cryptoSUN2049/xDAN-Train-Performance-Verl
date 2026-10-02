#!/usr/bin/env bash
# Network-volume cleanup, approved by the user on 2026-10-02.
# usage: cleanup.sh dry|run
set -uo pipefail
MODE="${1:-dry}"
W=/workspace
LOG=/root/cleanup-audit/cleanup-$(date -u +%Y%m%dT%H%M%SZ)-$MODE.log
TARGETS=/root/cleanup-audit/targets-$MODE.txt
: > "$TARGETS"

add() { for p in "$@"; do [ -e "$p" ] && echo "$p" >> "$TARGETS"; done; }

# --- A. ms-swift-jev: keep data and code; drop model weights, HF model caches, venvs ---
find "$W/ms-swift-jev" -xdev -type f \( -name "*.safetensors" -o -name "*.gguf" -o -name "pytorch_model*.bin" \
  -o -name "model*.bin" -o -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.onnx" -o -name "*.distcp" \) >> "$TARGETS"
find "$W/ms-swift-jev" -xdev -maxdepth 5 -type d \( -name "models--*" -o -name "hf-cache*" -o -name "hf-home" \
  -o -name venv -o -name envs -o -name .venv -o -path "*/cache/huggingface" -o -path "*/.cache/huggingface" \) -prune >> "$TARGETS"

# --- B. train-p0-dsh-integration (ours) ---
add "$W/train-p0-dsh-integration/backups/fullgpu-eu-64k-r2-global_step_1.tar" \
    "$W/train-p0-dsh-integration/backups/fullgpu-eu-64k-r2-global_step_2.tar" \
    "$W/train-p0-dsh-integration/runs/fullgpu-r3/checkpoints" \
    "$W/train-p0-dsh-integration/restored-is1-r1/global_step_2" \
    "$W/train-p0-dsh-integration/runs/fullgpu-train8-separate-is1-r2/checkpoints/global_step_4" \
    "$W/train-p0-dsh-integration/checkpoints/official-code-batch2-mode2-4gpu-20261002-r2/global_step_1" \
    "$W/train-p0-dsh-integration/checkpoints/official-music-4gpu-20261002-m1/global_step_2"

# --- C. verl-uni-agent-harbor-opd-rl (other team; user-approved) ---
V="$W/verl-uni-agent-harbor-opd-rl"
for r in pipe-r1 pipe-r2 pipe-r3 pipe-r4; do
  find "$V/runs/$r" -xdev -type d \( -name checkpoints -o -name pinned \) -prune >> "$TARGETS" 2>/dev/null
done
# SFT: keep only 64k-20k-fsdp2 global_step_10000
find "$V/runs/performance-9b-sft" -xdev -type d -name "global_step_*" -prune \
  ! -path "*verl-sft-cu128-64k-20k-fsdp2-20260925T0756Z/checkpoints/global_step_10000" >> "$TARGETS"
# Eval: keep sft-final-step10000/merged-model; drop duplicate merger copy and step5000 merges
add "$V/runs/performance-9b-eval/sft-final-step10000/merged-model.verl-merger"
find "$V/runs/performance-9b-eval" -mindepth 2 -maxdepth 2 -type d -path "*/sft-step5000-*/*" \
  \( -name "merged-model*" -o -name huggingface \) >> "$TARGETS"
add "$V/cache/uv"

# --- D-H. old standalone items ---
add "$W/skyrl" "$W/download" "$W/.venvs" "$W/models/Qwen3.5-9B" "$W/models/Qwen3-4B-1cfa9a7"
for d in router-semantic-ab-release-v3 native-candidate-complete-r1 github-plugin-assembled-r1; do
  find "$W/$d" -xdev -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \) -size +100M >> "$TARGETS"
done

# Hard guard: never touch these
GUARD='^/workspace/(env_infra|models/MiMo-V2.6-Distill-Qwen-9B|models/Qwen3.8-27B|mimo-dsh-rl-20260928|datasets|apus-data-cleaning|envs$|envs/)|checkpoints/official-(cyber|general|webdev)|official-code-batch2-mode2-4gpu-20261002-r2/global_step_2|official-music-4gpu-20261002-m1/global_step_3|global_step_10000$|sft-final-step10000/merged-model$|/runs/pipe-(r11|s1|s2)'
sort -u "$TARGETS" -o "$TARGETS"
if grep -E "$GUARD" "$TARGETS"; then echo "GUARD HIT - aborting"; exit 2; fi

total=0
while IFS= read -r p; do
  s=$(du -sb "$p" 2>/dev/null | cut -f1); s=${s:-0}; total=$((total + s))
  printf "%s\t%s\n" "$s" "$p" >> "$LOG"
done < "$TARGETS"
echo "targets: $(wc -l < "$TARGETS")  total: $((total / 1000000000)) GB" | tee -a "$LOG"

if [ "$MODE" = run ]; then
  while IFS= read -r p; do rm -rf -- "$p" && echo "deleted $p" >> "$LOG"; done < "$TARGETS"
  echo "done; log $LOG"
fi
