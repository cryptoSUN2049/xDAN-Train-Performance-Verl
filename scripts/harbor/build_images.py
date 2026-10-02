# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Build Harbor task images on Modal from each task's environment/Dockerfile.

Writes an image map ``{task: {"image": "im-...", "env_sha256": ..., ...}}`` that
``prepare_data.py --image-map`` folds into the rows; MiMo-Agent's Modal backend runs ``im-`` ids
directly (``modal.Image.from_id``). The map is keyed by a hash of the whole build context, so
re-running skips tasks that are already built and rebuilds tasks whose environment changed.

usage: build_images.py --tasks-root DIR --map image-map.json [--task NAME ...] [--limit N] [--workers 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_APP = "xdan-harbor-image-build"


def context_sha256(environment_dir: Path) -> str:
    """Hash every file in the build context (path + bytes) so any change forces a rebuild."""
    digest = hashlib.sha256()
    for path in sorted(p for p in environment_dir.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(environment_dir)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_one(task_dir: Path, app) -> dict:
    import modal

    environment_dir = task_dir / "environment"
    started = time.time()
    image = modal.Image.from_dockerfile(
        path=str(environment_dir / "Dockerfile"),
        context_dir=str(environment_dir),
        force_build=False,
    )
    image.build(app)
    return {"image": image.object_id, "seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--task", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--app-name", default=DEFAULT_APP)
    args = parser.parse_args()

    import modal

    names = args.task or sorted(p.name for p in args.tasks_root.iterdir() if (p / "environment/Dockerfile").is_file())
    if args.limit:
        names = names[: args.limit]
    image_map: dict = json.loads(args.map.read_text()) if args.map.exists() else {}
    lock = threading.Lock()

    todo = []
    for name in names:
        task_dir = args.tasks_root / name
        sha = context_sha256(task_dir / "environment")
        entry = image_map.get(name, {})
        if entry.get("env_sha256") == sha and entry.get("image"):
            continue
        todo.append((name, task_dir, sha))
    print(json.dumps({"requested": len(names), "to_build": len(todo), "cached": len(names) - len(todo)}), flush=True)

    app = modal.App.lookup(args.app_name, create_if_missing=True)

    def save() -> None:
        tmp = args.map.with_suffix(".tmp")
        tmp.write_text(json.dumps(image_map, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, args.map)

    def work(item):
        name, task_dir, sha = item
        try:
            result = build_one(task_dir, app)
            record = {**result, "env_sha256": sha, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        except Exception as error:  # noqa: BLE001 - recorded per task, retried on the next run
            record = {"error": repr(error)[:1000], "env_sha256": sha}
        with lock:
            image_map[name] = record
            save()
        return name, record

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(work, item) for item in todo]):
            name, record = future.result()
            failures += "error" in record
            print(json.dumps({"task": name, **{k: record.get(k) for k in ("image", "seconds", "error")}}), flush=True)
    built = sum(1 for name in names if image_map.get(name, {}).get("image"))
    print(json.dumps({"built": built, "of": len(names), "failures": failures}), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
