"""Check Harbor task images for post-base git history (answer leak) in Modal sandboxes.

For each sampled task, starts its built image (im- id from image-map.json) with no network job, runs read-only
git queries in the task cwd and records how many commits are reachable beyond HEAD (other branches, tags,
reflog, packed future commits). Sandboxes run under their own Modal app and are terminated right away.

usage: git_leak_check.py --data train.parquet --image-map image-map.json [--prefix swe-rebench] [--n 5] [--seed 0]
"""

import argparse
import json
import random

import modal
import pandas as pd

QUERY = r"""
cd "$1" 2>/dev/null || { echo "NO_CWD"; exit 0; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "NO_GIT"; exit 0; }
echo "HEAD=$(git rev-parse --short HEAD)"
echo "beyond_head=$(git rev-list --all --not HEAD 2>/dev/null | wc -l)"
echo "refs=$(git for-each-ref --format='%(refname:short)' | head -20 | tr '\n' ' ')"
echo "reflog=$(git reflog 2>/dev/null | wc -l)"
echo "objects=$(git count-objects -v 2>/dev/null | tr '\n' ' ')"
echo "log_all_head=$(git log --all --oneline 2>/dev/null | head -3 | tr '\n' '|')"
echo "reachable_objects=$(git rev-list --objects --all 2>/dev/null | wc -l)"
echo "unreachable_commits=$(git fsck --unreachable --no-reflogs 2>/dev/null | grep -c 'unreachable commit')"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-map", required=True)
    parser.add_argument("--prefix", default="swe-rebench")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for info in pd.read_parquet(args.data)["extra_info"]:
        if str(info["instance_id"]).startswith(args.prefix):
            spec = json.loads(info["instance_json"])
            rows.append((str(info["instance_id"]), spec["docker_image"], spec.get("cwd") or "/"))
    random.Random(args.seed).shuffle(rows)
    app = modal.App.lookup("xdan-fusion-leakcheck", create_if_missing=True)
    for instance_id, image_id, cwd in rows[: args.n]:
        sandbox = modal.Sandbox.create(app=app, image=modal.Image.from_id(image_id), timeout=600, cpu=1, memory=2048)
        try:
            process = sandbox.exec("bash", "-c", QUERY, "_", cwd, timeout=120)
            out = process.stdout.read().strip().replace("\n", " ; ")
        finally:
            sandbox.terminate()
        print(json.dumps({"instance_id": instance_id, "cwd": cwd, "result": out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
