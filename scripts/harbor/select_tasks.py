# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Deterministically select a training batch from the Harbor full pool (xDAN-Harbor-Stage1-Tasks-Full).

Rules (mirroring the stage1 slices): only ``status == derived`` rows of the ``train`` split; never an
eval-set reservation or an already-used task; order by sha256(source/task) within each source; at most
``--max-per-repo`` SWE tasks per repository; per-source quotas. Writes the selected task directories
(relative to the pool root) plus the rules and hashes needed to reproduce the batch.

usage: select_tasks.py --pool-root DIR --quota swe-rebench-v2-fv=900 --quota terminal-lego-15k=1100
       [--exclude-names stage1-names.txt ...] --out selection.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def order_key(row: dict) -> str:
    return hashlib.sha256(f"{row['source']}/{row['task']}".encode()).hexdigest()


def eval_reservation(pool_root: Path) -> set[tuple[str, str]]:
    selected = json.loads((pool_root / "eval-set-v1" / "selected.json").read_text())
    return {(item["source"], item["task"]) for item in selected["tasks"]}


def select(rows: list[dict], quotas: dict[str, int], excluded: set[tuple[str, str]], max_per_repo: int) -> list[dict]:
    chosen: list[dict] = []
    for source, quota in quotas.items():
        per_repo: Counter = Counter()
        candidates = sorted(
            (
                r
                for r in rows
                if r["source"] == source
                and r.get("status") == "derived"
                and r.get("split") == "train"
                and (r["source"], r["task"]) not in excluded
            ),
            key=order_key,
        )
        taken = 0
        for row in candidates:
            if taken >= quota:
                break
            repo = row.get("category") or ""
            if source.startswith("swe") and per_repo[repo] >= max_per_repo:
                continue
            per_repo[repo] += 1
            chosen.append(row)
            taken += 1
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--quota", action="append", required=True, help="source=count")
    parser.add_argument("--exclude-names", type=Path, action="append", default=[], help="lines of source/task")
    parser.add_argument("--max-per-repo", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    index_path = args.pool_root / "index" / "tasks.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    excluded = eval_reservation(args.pool_root)
    reserved = len(excluded)
    for path in args.exclude_names:
        # Accepts the evaluation deny list format: comments and bare names (no source) are ignored here.
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "/" in line:
                source, task = line.split("/", 1)
                excluded.add((source, task))
    quotas = {}
    for item in args.quota:
        source, count = item.split("=")
        quotas[source] = int(count)
    chosen = select(rows, quotas, excluded, args.max_per_repo)
    stats = defaultdict(Counter)
    for row in chosen:
        stats[row["source"]][row.get("difficulty")] += 1
    result = {
        "pool": str(args.pool_root),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "rules": {
            "status": "derived",
            "split": "train",
            "order": "sha256(source/task) within source",
            "max_per_repo_swe": args.max_per_repo,
            "quotas": quotas,
            "eval_reservation_excluded": reserved,
            "extra_excluded": len(excluded) - reserved,
        },
        "counts": {source: dict(counter) for source, counter in stats.items()},
        "tasks": [{"source": r["source"], "task": r["task"], "task_dir": r["task_dir"]} for r in chosen],
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"selected": len(chosen), "counts": result["counts"], "rules": result["rules"]}))


if __name__ == "__main__":
    main()
