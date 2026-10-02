#!/usr/bin/env bash
# SFT-baseline eval sets for group A and a model-free DSH payload check on official Code images.
# usage: prepare_eval_and_checks.sh <source-dir>
set -uo pipefail
S="${1:?source dir}"
P=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
source "$P/activate.sh" "$P" >/dev/null
source /workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/scripts/runtime.env
export PYTHONPATH="$S:$S/third_party/mimoagent-osr/src:$S/third_party/uni_agent"
D=/workspace/xdan-verl-fusion/data
E=$D/eval-a
mkdir -p "$E"
cd "$S"

# Eval rows are built WITHOUT the deny list on purpose: these are the evaluation sets.
python scripts/harbor/prepare_data.py \
  --tasks-root /workspace/verl-uni-agent-harbor-opd-rl/data/harbor/terminal-bench_terminal-bench-2-1/terminal-bench-2-1 \
  --out "$E/tb21.parquet"
cp /workspace/train-p0-dsh-integration/data-tiers-20261002/code/holdout100.parquet "$E/code-holdout100.parquet"
python - <<EOF
import pandas as pd
tb = pd.read_parquet("$E/tb21.parquet")
code = pd.read_parquet("$E/code-holdout100.parquet")
cols = sorted(set(tb.columns) & set(code.columns))
both = pd.concat([tb[cols], code[cols]], ignore_index=True)
both.to_parquet("$E/tb21+code-holdout100.parquet", index=False)
print({"tb21": len(tb), "code_holdout": len(code), "combined": len(both), "columns": cols})
EOF

# DSH payload on official Code images (4 holdout rows; model-free keyless smoke only, no grading).
python - <<EOF
import pandas as pd
pd.read_parquet("$E/code-holdout100.parquet").head(4).to_parquet("$E/code-dsh-check4.parquet", index=False)
EOF
python scripts/harbor/dsh_runtime_check.py --data "$E/code-dsh-check4.parquet" \
  --harness config/agent/mixed/dsh-sdk-modal.yaml --out "$E/code-dsh-check4.json"
echo "dsh check rc=$?"
