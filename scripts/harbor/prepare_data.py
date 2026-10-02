# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Convert Harbor task directories into Code-recipe parquet rows (dataset_type: harbor).

Each row keeps the Code recipe contract (``extra_info.instance_json`` JSON string, prompt equal to the
task instruction), so any Code harness spec can run it. Tests travel inline as a gzipped tarball with
its SHA256 so the trainer host never needs the task directory at grading time.

usage: prepare_data.py --tasks-root DIR --task NAME [--task NAME ...] --out train.parquet
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
    """Harbor runs the verifier in the image WORKDIR; task.toml does not carry it."""
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return "/app"
    matches = _WORKDIR.findall(dockerfile.read_text())
    return matches[-1] if matches else "/app"


def tests_tarball(task_dir: Path) -> bytes:
    tests = task_dir / "tests"
    if not (tests / "test.sh").is_file():
        raise ValueError(f"{task_dir.name}: tests/test.sh missing")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(tests.rglob("*")):
            archive.add(path, arcname=str(path.relative_to(tests)), recursive=False)
    return buffer.getvalue()


def task_row(task_dir: Path, index: int) -> dict:
    config = tomllib.loads((task_dir / "task.toml").read_text())
    environment = config.get("environment", {})
    image = environment.get("docker_image")
    if not image:
        raise ValueError(f"{task_dir.name}: no prebuilt docker_image (Dockerfile builds are not supported yet)")
    instruction = (task_dir / "instruction.md").read_text()
    if not instruction.strip():
        raise ValueError(f"{task_dir.name}: empty instruction.md")
    tests = tests_tarball(task_dir)
    if len(tests) > MAX_INLINE_TESTS_BYTES:
        raise ValueError(f"{task_dir.name}: tests tarball {len(tests)} bytes exceeds inline limit")
    instance = {
        "dataset_type": DATASET_TYPE,
        "instance_id": task_dir.name,
        "docker_image": image,
        "cwd": dockerfile_workdir(task_dir),
        "problem_statement": instruction,
        "verifier_timeout_sec": float(config.get("verifier", {}).get("timeout_sec", 900.0)),
        "agent_timeout_sec": float(config.get("agent", {}).get("timeout_sec", 900.0)),
        "cpus": environment.get("cpus"),
        "memory_mb": environment.get("memory_mb"),
        "allow_internet": bool(environment.get("allow_internet", True)),
        "tests_tar_b64": base64.b64encode(tests).decode(),
        "tests_sha256": hashlib.sha256(tests).hexdigest(),
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
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import pandas as pd

    rows = [task_row(args.tasks_root / name, index) for index, name in enumerate(args.task)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    manifest = {
        "tasks_root": str(args.tasks_root),
        "tasks": [
            {
                "instance_id": row["extra_info"]["instance_id"],
                **{
                    key: json.loads(row["extra_info"]["instance_json"])[key]
                    for key in ("docker_image", "cwd", "verifier_timeout_sec", "tests_sha256")
                },
            }
            for row in rows
        ],
    }
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
