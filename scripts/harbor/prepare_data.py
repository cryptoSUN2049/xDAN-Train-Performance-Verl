# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Convert Harbor task directories into Code-recipe parquet rows (dataset_type: harbor).

Each row keeps the Code recipe contract (``extra_info.instance_json`` JSON string, prompt equal to the
task instruction), so any Code harness spec can run it. Tests travel inline as a gzipped tarball with
its SHA256 so the trainer host never needs the task directory at grading time.

usage: prepare_data.py --tasks-root DIR [--task NAME ...] [--image-map map.json [--skip-unbuilt]] --out train.parquet
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import tomllib

DATASET_TYPE = "harbor"
MAX_INLINE_TESTS_BYTES = 2 * 1024 * 1024
_WORKDIR = re.compile(r"^\s*WORKDIR\s+(\S+)\s*$", re.IGNORECASE | re.MULTILINE)


def dockerfile_workdir(task_dir: Path) -> str:
    """Fallback: the image WORKDIR (last one wins), which is where Harbor runs the verifier."""
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return "/app"
    matches = _WORKDIR.findall(dockerfile.read_text())
    return matches[-1] if matches else "/app"


def task_workdir(task_dir: Path, config: dict) -> str:
    """Prefer the task's declared ``[harbor_runtime] workdir``; fall back to the Dockerfile WORKDIR."""
    return config.get("harbor_runtime", {}).get("workdir") or dockerfile_workdir(task_dir)


def resolve_image(task_dir: Path, environment: dict, image_map: dict | None) -> str:
    """Prebuilt ``docker_image`` from task.toml, else the Modal-built ``im-`` id for this exact context."""
    if environment.get("docker_image"):
        return environment["docker_image"]
    entry = (image_map or {}).get(task_dir.name) or {}
    if not entry.get("image"):
        raise ValueError(f"{task_dir.name}: no prebuilt docker_image and no built image (run build_images.py)")
    from scripts.harbor.build_images import context_sha256

    if entry.get("env_sha256") != context_sha256(task_dir / "environment"):
        raise ValueError(f"{task_dir.name}: built image is stale; environment changed since build")
    return entry["image"]


def task_identities(name: str) -> set[str]:
    """Names a task may be listed under: its directory name and, for ``[NNNN__]source__task``, ``source/task``."""
    identities = {name}
    parts = name.split("__")
    if len(parts) >= 2:
        if parts[0].isdigit():
            parts = parts[1:]
        identities.add(f"{parts[0]}/{'__'.join(parts[1:])}")
        identities.add("__".join(parts[1:]))
    return identities


def load_deny_list(paths: list[Path]) -> set[str]:
    denied: set[str] = set()
    for path in paths:
        denied.update(
            line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")
        )
    return denied


def tests_tarball(task_dir: Path) -> bytes:
    tests = task_dir / "tests"
    if not (tests / "test.sh").is_file():
        raise ValueError(f"{task_dir.name}: tests/test.sh missing")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(tests.rglob("*")):
            archive.add(path, arcname=str(path.relative_to(tests)), recursive=False)
    return buffer.getvalue()


def task_row(task_dir: Path, index: int, image_map: dict | None = None) -> dict:
    config = tomllib.loads((task_dir / "task.toml").read_text())
    environment = config.get("environment", {})
    image = resolve_image(task_dir, environment, image_map)
    instruction = (task_dir / "instruction.md").read_text()
    if not instruction.strip():
        raise ValueError(f"{task_dir.name}: empty instruction.md")
    tests = tests_tarball(task_dir)
    if len(tests) > MAX_INLINE_TESTS_BYTES:
        # Large test suites stay on the shared volume; the environment re-checks this digest at grading.
        from recipes.harbor.environment import tests_dir_digest

        test_fields = {
            "task_path": str(task_dir.resolve()),
            "tests_dir_sha256": tests_dir_digest(str(task_dir / "tests")),
        }
    else:
        test_fields = {
            "tests_tar_b64": base64.b64encode(tests).decode(),
            "tests_sha256": hashlib.sha256(tests).hexdigest(),
        }
    instance = {
        "dataset_type": DATASET_TYPE,
        "instance_id": task_dir.name,
        "docker_image": image,
        "cwd": task_workdir(task_dir, config),
        "problem_statement": instruction,
        "verifier_timeout_sec": float(config.get("verifier", {}).get("timeout_sec", 900.0)),
        "agent_timeout_sec": float(config.get("agent", {}).get("timeout_sec", 900.0)),
        "cpus": environment.get("cpus"),
        "memory_mb": environment.get("memory_mb"),
        "allow_internet": bool(environment.get("allow_internet", True)),
        **test_fields,
    }
    return {
        "data_source": DATASET_TYPE,
        "ability": "agentic",
        "agent_name": "mimo_swe_agent",
        "prompt": [{"role": "user", "content": instruction}],
        "reward_model": {"ground_truth": "", "style": "rule"},
        "extra_info": {
            "dataset_type": DATASET_TYPE,
            "index": index,
            "instance_id": task_dir.name,
            "instance_json": json.dumps(instance, sort_keys=True),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--task", action="append", help="default: every task directory under --tasks-root")
    parser.add_argument("--image-map", type=Path, help="build_images.py output for tasks without docker_image")
    parser.add_argument("--skip-unbuilt", action="store_true", help="skip (and report) tasks without an image")
    parser.add_argument(
        "--deny-list",
        type=Path,
        action="append",
        default=[],
        help="evaluation/holdout task identities (one per line); any hit aborts the conversion",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    names = args.task or sorted(p.name for p in args.tasks_root.iterdir() if (p / "task.toml").is_file())
    denied = load_deny_list(args.deny_list)
    leaked = sorted(name for name in names if task_identities(name) & denied)
    if leaked:
        raise SystemExit(f"{len(leaked)} evaluation/holdout tasks in the training input, e.g. {leaked[:5]}")
    image_map = json.loads(args.image_map.read_text()) if args.image_map else None
    rows, skipped = [], {}
    for name in names:
        try:
            rows.append(task_row(args.tasks_root / name, len(rows), image_map))
        except ValueError as error:
            if not args.skip_unbuilt:
                raise
            skipped[name] = str(error)
    if not rows:
        raise SystemExit("no rows produced")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    manifest = {
        "tasks_root": str(args.tasks_root),
        "tasks": [
            {
                "instance_id": row["extra_info"]["instance_id"],
                **{
                    key: json.loads(row["extra_info"]["instance_json"]).get(key)
                    for key in ("docker_image", "cwd", "verifier_timeout_sec", "tests_sha256", "tests_dir_sha256")
                },
            }
            for row in rows
        ],
        "skipped": skipped,
        "deny_lists": [str(path) for path in args.deny_list],
        "denied_identities": len(denied),
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "skipped": len(skipped), "out": str(args.out)}))


if __name__ == "__main__":
    main()
