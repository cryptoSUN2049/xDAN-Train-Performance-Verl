# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Combine Code-recipe parquets (MiMo Code + Harbor rows) into one training set, enforcing the deny list.

Every row keeps its own ``dataset_type`` inside ``extra_info.instance_json``; the runner dispatches per
row. The deny list is re-checked on the combined set (instance ids and ``source/task`` identities) so an
evaluation task can never reach training through any input.

usage: combine_group_a.py --input a.parquet --input b.parquet --deny-list eval-denylist.txt --out train.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.harbor.prepare_data import load_deny_list, task_identities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--deny-list", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    frames, per_input = [], {}
    for path in args.input:
        frame = pd.read_parquet(path)
        per_input[str(path)] = {"rows": len(frame), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        frames.append(frame)
    columns = sorted(set.intersection(*(set(frame.columns) for frame in frames)))
    combined = pd.concat([frame[columns] for frame in frames], ignore_index=True)

    denied = load_deny_list(args.deny_list)
    ids = [str(info["instance_id"]) for info in combined["extra_info"]]
    leaked = sorted(i for i in ids if task_identities(i) & denied)
    if leaked:
        raise SystemExit(f"{len(leaked)} evaluation/holdout tasks in training input, e.g. {leaked[:5]}")
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate instance ids across inputs")

    combined = combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.out, index=False)
    types = combined["extra_info"].map(lambda info: json.loads(info["instance_json"])["dataset_type"])
    manifest = {
        "inputs": per_input,
        "deny_lists": [str(p) for p in args.deny_list],
        "rows": len(combined),
        "by_dataset_type": types.value_counts().to_dict(),
        "columns": columns,
        "shuffle_seed": args.seed,
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "by_dataset_type")}))


if __name__ == "__main__":
    main()
