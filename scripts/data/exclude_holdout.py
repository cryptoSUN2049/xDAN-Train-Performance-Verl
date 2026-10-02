# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Write a training parquet with every holdout instance removed.

The official Code train set (2698 rows) contains all 100 holdout tasks, so training on it and
evaluating on the holdout leaks. Rows are matched on ``extra_info.instance_id``; the manifest
records both input hashes and the removed ids so the split can be audited and reproduced.

usage: exclude_holdout.py --train train.parquet --holdout holdout.parquet [--holdout ...] --out clean.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def instance_ids(frame) -> list[str]:
    return [str(info["instance_id"]) for info in frame["extra_info"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    train = pd.read_parquet(args.train)
    held = set()
    for path in args.holdout:
        held.update(instance_ids(pd.read_parquet(path)))
    ids = instance_ids(train)
    keep = [identity not in held for identity in ids]
    clean = train[keep].reset_index(drop=True)
    removed = sorted(identity for identity in ids if identity in held)
    if len(removed) != len(held):
        missing = sorted(held - set(ids))
        print(json.dumps({"warning": "holdout ids absent from train", "count": len(missing), "examples": missing[:5]}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(args.out, index=False)
    manifest = {
        "train": str(args.train),
        "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
        "holdouts": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.holdout},
        "rows_in": len(train),
        "rows_out": len(clean),
        "removed": removed,
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows_in", "rows_out")} | {"removed": len(removed)}))


if __name__ == "__main__":
    main()
