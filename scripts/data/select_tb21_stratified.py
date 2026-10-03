# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Pick a Harbor batch1 subset whose category mix follows the TB2.1 category distribution.

``select`` (no third-party deps) reads the TB2.1-taxonomy labels written by ``relabel_tb21_taxonomy.py`` and
writes a selection JSON with two variants:

* ``a`` strict-proportional: categories without batch1 coverage are dropped, the remaining TB2.1 shares are
  renormalised, and N is the largest total whose (largest-remainder) quotas all fit the availability.
* ``b`` cap-and-fill: ``--total-b`` slots are apportioned over all 16 TB2.1 shares, each quota is capped at
  availability, and the shortfall is water-filled into covered categories weighted by TB2.1 share.

Within a category, tasks are ranked medium/hard before easy (native difficulty and LLM ``difficulty_hint``),
ties broken by a seeded shuffle. SWE repos are capped at ``--max-per-repo`` tasks before availability is
counted, so the cap never has to be enforced after quotas are fixed.

``materialize`` (pod, needs pandas) subsets batch1 ``train.parquet`` to one variant, runs
``combine_group_a.py`` with MiMo Code (deny list, duplicate check, seeded shuffle), and extends its manifest
with output sha256, per-source/category/difficulty counts and the selected ids.

usage:
  select_tb21_stratified.py select --labels labels.jsonl --out selection.json [--total-b 500]
  select_tb21_stratified.py materialize --selection selection.json --variant a --batch1 train.parquet \
      --code code.parquet --deny-list eval-denylist.txt --out-dir data/group-a-r2
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

# TB2.1 (89 tasks) ``metadata.category`` counts, from data/eval-a/tb21.manifest.json task.toml files (2026-10-03).
TB21_COUNTS: dict[str, int] = {
    "software-engineering": 26,
    "system-administration": 9,
    "scientific-computing": 8,
    "security": 8,
    "data-science": 8,
    "debugging": 5,
    "file-operations": 5,
    "model-training": 4,
    "mathematics": 4,
    "data-processing": 4,
    "machine-learning": 3,
    "games": 1,
    "personal-assistant": 1,
    "optimization": 1,
    "data-querying": 1,
    "video-processing": 1,
}
TB21_DIFFICULTY = {"easy": 4, "medium": 55, "hard": 30}
HARD_ENOUGH = {"medium", "hard"}


def repo_of(instance_id: str) -> str | None:
    """``swe-rebench-v2-fv__owner__repo-123`` -> ``owner__repo``; non-SWE tasks have no repo."""
    family, _, rest = instance_id.partition("__")
    if not family.startswith("swe-"):
        return None
    return rest.rsplit("-", 1)[0]


def priority(row: dict) -> int:
    """0: native and LLM both medium/hard; 1: one of them; 2: both easy."""
    return 2 - (row["native_difficulty"] in HARD_ENOUGH) - (row["difficulty_hint"] in HARD_ENOUGH)


def apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder apportionment; ties go to the larger weight, then the name."""
    weight_sum = sum(weights.values())
    if total <= 0 or weight_sum <= 0:
        return dict.fromkeys(weights, 0)
    exact = {k: total * w / weight_sum for k, w in weights.items()}
    quotas = {k: int(v) for k, v in exact.items()}
    order = sorted(weights, key=lambda k: (-(exact[k] - quotas[k]), -weights[k], k))
    for k in order[: total - sum(quotas.values())]:
        quotas[k] += 1
    return quotas


def ranked_pools(labels: list[dict], seed: int, max_per_repo: int) -> tuple[dict[str, list[dict]], dict]:
    """Per-category task lists in pick order, after the per-repo cap; also returns cap statistics."""
    rng = random.Random(seed)
    rows = sorted(labels, key=lambda r: r["instance_id"])
    rng.shuffle(rows)
    rows.sort(key=priority)  # stable: shuffle order survives within a priority tier
    per_repo: collections.Counter = collections.Counter()
    kept, dropped = [], collections.Counter()
    for row in rows:
        repo = repo_of(row["instance_id"])
        if repo is not None:
            if per_repo[repo] >= max_per_repo:
                dropped[repo] += 1
                continue
            per_repo[repo] += 1
        kept.append(row)
    pools: dict[str, list[dict]] = {c: [] for c in TB21_COUNTS}
    for row in kept:
        pools[row["category"]].append(row)
    return pools, {"repos_capped": dict(dropped), "tasks_dropped_by_cap": sum(dropped.values())}


def strict_proportional(avail: dict[str, int]) -> dict[str, int]:
    shares = {c: n for c, n in TB21_COUNTS.items() if avail[c] > 0}
    best = dict.fromkeys(TB21_COUNTS, 0)
    for total in range(1, sum(avail[c] for c in shares) + 1):
        quotas = apportion(total, shares)
        if all(quotas[c] <= avail[c] for c in shares):
            best = {c: quotas.get(c, 0) for c in TB21_COUNTS}
    return best


def cap_and_fill(avail: dict[str, int], total: int) -> dict[str, int]:
    total = min(total, sum(avail.values()))
    alloc = {c: min(q, avail[c]) for c, q in apportion(total, TB21_COUNTS).items()}
    while (remaining := total - sum(alloc.values())) > 0:
        open_cats = {c: TB21_COUNTS[c] for c in TB21_COUNTS if alloc[c] < avail[c]}
        if not open_cats:
            break
        for c, extra in apportion(remaining, open_cats).items():
            alloc[c] = min(avail[c], alloc[c] + extra)
    return alloc


def describe(chosen: list[dict], quotas: dict[str, int], avail: dict[str, int]) -> dict:
    n = len(chosen)
    tb_total = sum(TB21_COUNTS.values())
    tv = 0.5 * sum(abs(quotas[c] / max(n, 1) - TB21_COUNTS[c] / tb_total) for c in TB21_COUNTS)
    repos = collections.Counter(repo_of(r["instance_id"]) for r in chosen if repo_of(r["instance_id"]))
    return {
        "n": n,
        "tv_distance_to_tb21": round(tv, 4),
        "per_category": {
            c: {
                "tb21": TB21_COUNTS[c],
                "tb21_share": round(TB21_COUNTS[c] / tb_total, 4),
                "available": avail[c],
                "selected": quotas[c],
                "selected_share": round(quotas[c] / max(n, 1), 4),
            }
            for c in TB21_COUNTS
        },
        "by_family": dict(collections.Counter(r["family"] for r in chosen)),
        "by_native_difficulty": dict(collections.Counter(r["native_difficulty"] for r in chosen)),
        "by_llm_difficulty": dict(collections.Counter(r["difficulty_hint"] for r in chosen)),
        "by_priority_tier": dict(collections.Counter(priority(r) for r in chosen)),
        "max_tasks_per_repo": max(repos.values(), default=0),
        "ids": sorted(r["instance_id"] for r in chosen),
        "labels": {
            r["instance_id"]: {
                "category": r["category"],
                "native_difficulty": r["native_difficulty"],
                "difficulty_hint": r["difficulty_hint"],
                "family": r["family"],
            }
            for r in sorted(chosen, key=lambda r: r["instance_id"])
        },
    }


def cmd_select(args: argparse.Namespace) -> None:
    labels = [json.loads(line) for line in args.labels.read_text().splitlines() if line.strip()]
    if len({r["instance_id"] for r in labels}) != len(labels):
        raise SystemExit("duplicate instance ids in labels")
    pools, cap_stats = ranked_pools(labels, args.seed, args.max_per_repo)
    avail = {c: len(pools[c]) for c in TB21_COUNTS}
    variants = {
        "a": ("strict-proportional", strict_proportional(avail)),
        "b": (f"cap-and-fill(total={args.total_b})", cap_and_fill(avail, args.total_b)),
    }
    out = {
        "labels": str(args.labels),
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "seed": args.seed,
        "max_per_repo": args.max_per_repo,
        "tb21_counts": TB21_COUNTS,
        "tb21_difficulty": TB21_DIFFICULTY,
        "labelled": len(labels),
        "availability": avail,
        "repo_cap": cap_stats,
        "variants": {},
    }
    for key, (name, quotas) in variants.items():
        chosen = [row for c in TB21_COUNTS for row in pools[c][: quotas[c]]]
        out["variants"][key] = {"method": name, **describe(chosen, quotas, avail)}
    sweep = {}
    for total in sorted({out["variants"]["a"]["n"], 300, 500, 800, 1000, 1500}):
        quotas = cap_and_fill(avail, total)
        chosen = [row for c in TB21_COUNTS for row in pools[c][: quotas[c]]]
        info = describe(chosen, quotas, avail)
        sweep[str(total)] = {k: info[k] for k in ("n", "tv_distance_to_tb21", "by_family", "by_priority_tier")}
    out["cap_and_fill_sweep"] = sweep
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{'category':24s} {'tb21':>5s} {'avail':>6s} {'a':>5s} {'b':>5s}")
    for c in TB21_COUNTS:
        row = [out["variants"][k]["per_category"][c]["selected"] for k in ("a", "b")]
        print(f"{c:24s} {TB21_COUNTS[c]:5d} {avail[c]:6d} {row[0]:5d} {row[1]:5d}")
    for key, variant in out["variants"].items():
        print(key, variant["method"], "n=", variant["n"], "tv=", variant["tv_distance_to_tb21"], variant["by_family"])
    print("sweep", json.dumps({k: (v["n"], v["tv_distance_to_tb21"]) for k, v in sweep.items()}))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_materialize(args: argparse.Namespace) -> None:
    import pandas as pd

    selection = json.loads(args.selection.read_text())
    variant = selection["variants"][args.variant]
    ids = set(variant["ids"])
    batch1 = pd.read_parquet(args.batch1)
    batch1_ids = batch1["extra_info"].map(lambda info: str(info["instance_id"]))
    subset = batch1[batch1_ids.isin(ids)].reset_index(drop=True)
    missing = ids - set(batch1_ids)
    if missing or len(subset) != len(ids):
        raise SystemExit(f"selection/batch1 mismatch: missing={sorted(missing)[:5]} rows={len(subset)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.out_dir / f"batch1-tb21-{args.variant}.parquet"
    subset.to_parquet(subset_path, index=False)

    out_path = args.out_dir / f"train-r2-{args.variant}.parquet"
    command = [sys.executable, "-m", "scripts.data.combine_group_a", "--input", str(args.code)]
    command += ["--input", str(subset_path), "--out", str(out_path), "--seed", str(args.seed)]
    for deny in args.deny_list:
        command += ["--deny-list", str(deny)]
    subprocess.run(command, check=True)

    manifest_path = out_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    combined = pd.read_parquet(out_path)
    labels = variant["labels"]
    batch1_rows = [labels[str(i["instance_id"])] for i in combined["extra_info"] if str(i["instance_id"]) in labels]
    manifest.update(
        {
            "output": str(out_path),
            "output_sha256": sha256(out_path),
            "variant": args.variant,
            "method": variant["method"],
            "selection": str(args.selection),
            "selection_sha256": sha256(args.selection),
            "labels_sha256": selection["labels_sha256"],
            "rows_by_source": {"mimo-code": len(combined) - len(batch1_rows), "harbor-batch1": len(batch1_rows)},
            "batch1_by_category": dict(collections.Counter(r["category"] for r in batch1_rows)),
            "batch1_by_family": dict(collections.Counter(r["family"] for r in batch1_rows)),
            "batch1_by_native_difficulty": dict(collections.Counter(r["native_difficulty"] for r in batch1_rows)),
            "batch1_by_llm_difficulty": dict(collections.Counter(r["difficulty_hint"] for r in batch1_rows)),
            "batch1_selected_ids": sorted(labels),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("output", "rows", "rows_by_source", "output_sha256")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    select = sub.add_parser("select")
    select.add_argument("--labels", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    select.add_argument("--seed", type=int, default=20261004)
    select.add_argument("--max-per-repo", type=int, default=30)
    select.add_argument("--total-b", type=int, default=500)
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--selection", type=Path, required=True)
    materialize.add_argument("--variant", choices=("a", "b"), required=True)
    materialize.add_argument("--batch1", type=Path, required=True)
    materialize.add_argument("--code", type=Path, required=True)
    materialize.add_argument("--deny-list", type=Path, action="append", required=True)
    materialize.add_argument("--out-dir", type=Path, required=True)
    materialize.add_argument("--seed", type=int, default=20261002, help="combine shuffle seed (as r1)")
    args = parser.parse_args()
    {"select": cmd_select, "materialize": cmd_materialize}[args.cmd](args)


if __name__ == "__main__":
    main()
