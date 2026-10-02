#!/usr/bin/env bash
# Full-pool batch 1: select 2000 train tasks, build images on Modal, audit nop=0/oracle=1, keep passers.
# Resumable: each step skips work already recorded (image map, audit JSONL).
set -uo pipefail
P=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
source "$P/activate.sh" "$P" >/dev/null
source /workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/scripts/runtime.env
S=/workspace/xdan-verl-fusion/source-c9a699ad
export PYTHONPATH="$S:$S/third_party/mimoagent-osr/src:$S/third_party/uni_agent"
F=/workspace/xdan-verl-fusion/data/harbor-full
B=$F/batch1
DENY=/workspace/xdan-verl-fusion/data/eval-denylist.txt
log() { echo "$(date -u +%FT%TZ) $*"; }
mkdir -p "$B"
cd "$S"

log "wait for extraction"
while pgrep -f "tar -xzf /workspace/verl-uni-agent-harbor-opd-rl/data-eval-set-v1" >/dev/null; do sleep 30; done
ls "$F/pool/terminal-lego-15k-full/"*/runtime-v1 | wc -l
ls "$F/pool/swe-rebench-v2-fv-full/"*/runtime-v1 | wc -l

log "stage1-used list"
ls /workspace/verl-uni-agent-harbor-opd-rl/data-pipe-s1/stage1/tasks-train /workspace/verl-uni-agent-harbor-opd-rl/data-pipe-s1/stage1/tasks-validation \
  | grep "__" | python -c '
import sys
for name in sys.stdin:
    parts = name.strip().split("__")
    print(f"{parts[1]}/{\"__\".join(parts[2:])}")' | sort -u > "$F/stage1-used.txt"
wc -l < "$F/stage1-used.txt"

log "select"
python scripts/harbor/select_tasks.py --pool-root "$F/pool" \
  --quota swe-rebench-v2-fv=900 --quota terminal-lego-15k=1100 \
  --exclude-names "$F/stage1-used.txt" --exclude-names "$DENY" --out "$B/selection.json" || exit 1

log "flat task links"
mkdir -p "$B/tasks"
python - <<EOF
import json, os
sel = json.load(open("$B/selection.json"))
for item in sel["tasks"]:
    link = os.path.join("$B/tasks", f"{item['source']}__{item['task']}")
    if not os.path.lexists(link):
        os.symlink(os.path.join("$F/pool", item["task_dir"]), link)
print("links", len(sel["tasks"]))
EOF

log "build images"
python scripts/harbor/build_images.py --tasks-root "$B/tasks" --map "$B/image-map.json" --workers "${BUILD_WORKERS:-16}"
log "build done rc=$?"

log "prepare rows (deny list enforced)"
python scripts/harbor/prepare_data.py --tasks-root "$B/tasks" --image-map "$B/image-map.json" --skip-unbuilt \
  --deny-list "$DENY" --out "$B/candidates.parquet" || exit 1

log "audit"
python scripts/harbor/oracle_check.py --data "$B/candidates.parquet" --harness config/agent/harbor/mini-mimocode-modal.yaml \
  --tasks-root "$B/tasks" --jsonl "$B/audit.jsonl" --out "$B/audit-summary.json" --workers "${AUDIT_WORKERS:-24}"
log "audit done rc=$?"

log "keep passers"
python - <<EOF
import json, pandas as pd
summary = json.load(open("$B/audit-summary.json"))
passed = set(summary["passed_ids"])
frame = pd.read_parquet("$B/candidates.parquet")
keep = [info["instance_id"] in passed for info in frame["extra_info"]]
train = frame[keep].reset_index(drop=True)
train.to_parquet("$B/train.parquet", index=False)
print(json.dumps({"candidates": len(frame), "audit_passed": len(passed), "train_rows": len(train)}))
EOF
log "batch1 done"
