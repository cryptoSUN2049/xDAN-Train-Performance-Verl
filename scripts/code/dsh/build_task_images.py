"""Plan eight deterministic Code tasks; publishing requires --execute-build."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

SEED = 20260929
TRAIN = "format-code-task-001661"
HOLDOUT = "format-code-task-001457"
TOOL_HASHES = {
    "deployment/bootstrap/prepare_harbor_context.py": (
        "7ae533c687cf7fb133d1a5bfe1f8000e3fc01c77ee5afd98bbe5fdb74989907e"
    ),
    "deployment/harbor/mimo/remote_builder.py": "832ffb7a3404eef59464fc0dc40bd97b7499da7fad1b6bd8ccf96adb77e77e87",
    "deployment/harbor/mimo/prepare_context.py": "f7c65dc24f8fa48e8636c57b30fc52115ece55dda3bba8235093d1ec491e363a",
    "deployment/harbor/mimo/publish_binding.py": "6a8c511a346bc87ad0f0c3362c9f55918605dfaf9eb071ab493b10edfcfd2724",
    "deployment/harbor/mimo/install.py": "87697339d66bbd574ccb8a8ecc8f6d9e713b2f7c55468a83721c2a2e9ea3fab9",
    "deployment/harbor/mimo/smoke.py": "fcda6e915a81efe841c73f5da90545d5765fb32d54fa46b4f2d9a05772c58188",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(root, relative, expected):
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha(path) != expected:
        raise ValueError(f"Pinned input changed or escaped root: {relative}")
    return path


def select_tasks(ids):
    ids = list(ids)
    if len(ids) != len(set(ids)) or TRAIN not in ids or HOLDOUT not in ids:
        raise ValueError("Expected unique IDs and both fixed tasks")
    candidates = sorted(set(ids) - {TRAIN, HOLDOUT})
    if len(candidates) < 7:
        raise ValueError("Need seven additional distinct tasks")
    return [TRAIN, *random.Random(SEED).sample(candidates, 7)]


def data_module():
    spec = importlib.util.spec_from_file_location("image_plan_data", Path(__file__).with_name("prepare_data.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_tasks(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = data_module()
    source = data._pinned_bytes(args.source_parquet, data.SOURCE_SHA256)
    mapping_bytes = data._pinned_bytes(args.image_mapping, data.MAPPING_SHA256)
    mapping = data._mapping(mapping_bytes)
    # Parse the opaque instance container, extracting only public metadata. Never print test fields.
    rows = pq.read_table(pa.BufferReader(source), columns=["extra_info"], use_threads=False).to_pylist()
    tasks = {}
    for row in rows:
        instance = json.loads(row["extra_info"]["instance_json"])
        task = {key: instance[key] for key in ("instance_id", "docker_image", "cwd", "dataset_type")}
        key = task["instance_id"]
        if not data.TASK_ID.fullmatch(key) or key in tasks or task["dataset_type"] != "opensource-code":
            raise ValueError("Invalid or duplicate Code task ID")
        if not task["cwd"].startswith("/") or ".." in Path(task["cwd"]).parts:
            raise ValueError("Invalid task cwd")
        task["mapped_image"] = mapping[task["docker_image"]]
        tasks[key] = task
    return data, tasks


def commands(args, task, lock_path, python_sha, source_commit):
    root = args.output / task["instance_id"]
    original = task["original_image"]
    builder = [args.python, "-m", "deployment.harbor.mimo.remote_builder"]
    common = [
        "--original-image",
        original,
        "--cpu",
        "2",
        "--memory-mib",
        "8192",
        "--task-cwd",
        task["cwd"],
        "--app-name",
        "train8-image-builder",
    ]
    probe = [*builder, "probe", *common, "--output", str(root / "probe"), "--timeout-seconds", "180"]
    context = [
        args.python,
        "-m",
        "deployment.harbor.mimo.prepare_context",
        "--repo",
        str(args.source_repo),
        "--artifact-dir",
        str(args.release_dir),
        "--python-archive",
        str(args.context / "python.tar.gz"),
        "--python-sha256",
        python_sha,
        "--source-commit",
        source_commit,
        "--original-image",
        original,
        "--gateway-origin",
        "https://gateway.invalid",
        "--lock",
        str(lock_path),
        "--base-inspect",
        str(root / "probe" / "original-inspect.json"),
        "--output",
        str(root / "context"),
    ]
    build = [
        *builder,
        "build",
        *common,
        "--output",
        str(root / "build"),
        "--timeout-seconds",
        "600",
        "--context",
        str(root / "context"),
        "--image-tag",
        task["image_tag"],
        "--registry-user",
        args.registry_user,
        "--registry-token-file",
        str(args.registry_token_file),
    ]
    return {"probe": probe, "context": context, "build": build}


def prepare_plan(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Output must be new; interrupted runs are preserved")
    if not args.output.parent.is_dir():
        raise ValueError("Output parent must exist")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", args.tag_suffix):
        raise ValueError("Invalid tag suffix")
    if not re.fullmatch(r"ghcr\.io/[a-z0-9._-]+/[a-z0-9._-]+", args.registry_prefix):
        raise ValueError("Expected lowercase GHCR image prefix")
    data, tasks = read_tasks(args)
    selected = select_tasks(tasks)
    hashes = {name: sha(checked_file(args.builder_source, name, expected)) for name, expected in TOOL_HASHES.items()}
    manifest_path = args.context / "context-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        checked_file(args.context, name, expected)
    inputs = json.loads((args.context / "build-inputs.json").read_text())
    lock = inputs["lock"]
    release_hashes = {
        name: sha(checked_file(args.release_dir, name, expected)) for name, expected in lock["wheels"].items()
    }
    if inputs["source_commit"] != manifest["source_commit"]:
        raise ValueError("Source commit mismatch")
    # git-show validates exactly the source bytes prepare_context will consume.
    for name, expected in inputs["source_files"].items():
        content = subprocess.check_output(
            ["git", "-C", str(args.source_repo), "show", f"{manifest['source_commit']}:{name}"]
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Pinned source file changed")
    existing = json.loads(args.existing_bindings.read_text())
    instances = {key: (0, {"docker_image": task["docker_image"]}) for key, task in tasks.items()}
    mapping = {task["docker_image"]: task["mapped_image"] for task in tasks.values()}
    bindings = data._bindings(args.existing_bindings.read_bytes(), instances, mapping)
    if TRAIN not in bindings or HOLDOUT not in bindings:
        raise ValueError("Both existing train and holdout bindings are required")
    originals = json.loads(args.original_images.read_text()) if args.original_images else {}
    planned = []
    for key in selected + [HOLDOUT]:
        task = dict(tasks[key])
        task["split"] = "holdout" if key == HOLDOUT else "train"
        task["reuse"] = key in bindings
        original = originals.get(key, bindings.get(key, {}).get("original_image"))
        if original is not None:
            if (
                not data.DIGEST_IMAGE.fullmatch(original)
                or original.split("@", 1)[0] != task["mapped_image"].rsplit(":", 1)[0]
            ):
                raise ValueError(f"Original image does not match official repository: {key}")
            if key in bindings and original != bindings[key]["original_image"]:
                raise ValueError("Existing image binding cannot change")
        task["original_image"] = original
        task["image_tag"] = f"{args.registry_prefix}-{key.removeprefix('format-code-task-')}:{args.tag_suffix}"
        if not task["reuse"] and original:
            task["commands"] = commands(
                args,
                task,
                args.output / "release-lock.json",
                manifest["files"]["python.tar.gz"],
                manifest["source_commit"],
            )
        planned.append(task)
    missing = [task["instance_id"] for task in planned if not task["original_image"]]
    if args.execute_build and missing:
        raise ValueError("Missing original digests: " + ", ".join(missing))
    if args.execute_build and not args.registry_token_file:
        raise ValueError("Execution needs a private registry credential file path")
    plan = {
        "schema": "xdan.train8-image-plan.v1",
        "seed": SEED,
        "train_ids": selected,
        "holdout_ids": [HOLDOUT],
        "selection": "random.Random(seed).sample(sorted(all_ids - fixed_train - fixed_holdout), 7)",
        "status": "planned",
        "missing_original_digests": missing,
        "tasks": planned,
        "source_revision": data.SOURCE_REVISION,
        "sha256": {
            "source_parquet": sha(args.source_parquet),
            "image_mapping": sha(args.image_mapping),
            "existing_bindings": sha(args.existing_bindings),
            "builder_tools": hashes,
            "context_manifest": sha(manifest_path),
            "context_files": manifest["files"],
            "release_files": release_hashes,
            "source_files": inputs["source_files"],
            "original_images": sha(args.original_images) if args.original_images else None,
        },
        "paths": {key: str(getattr(args, key)) for key in ("builder_source", "source_repo", "context", "release_dir")},
        "source_commit": manifest["source_commit"],
        "results": [],
    }
    args.output.mkdir(mode=0o700)
    (args.output / "release-lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan, existing, data, instances, mapping


def execute_plan(args, plan, existing, data, instances, mapping):
    environment = dict(os.environ, PYTHONPATH=str(args.builder_source), PYTHONDONTWRITEBYTECODE="1")
    merged = {entry["instance_id"]: entry for entry in existing["bindings"]}
    for task in plan["tasks"]:
        result = {"instance_id": task["instance_id"], "status": "reused" if task["reuse"] else "starting", "stages": []}
        plan["results"].append(result)
        try:
            if not task["reuse"]:
                task_root = args.output / task["instance_id"]
                task_root.mkdir(mode=0o700)
                for stage, command in task["commands"].items():
                    # Builder owns resource deadlines and cleanup; do not kill its client mid-cleanup.
                    with (task_root / f"{stage}.log").open("x") as log:
                        completed = subprocess.run(
                            command,
                            cwd=args.builder_source,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    result["stages"].append({"stage": stage, "returncode": completed.returncode})
                    if completed.returncode:
                        raise RuntimeError(f"{stage} failed")
                status = json.loads((task_root / "build/status.json").read_text())
                binding = json.loads((task_root / "build/image-binding.json").read_text())
                if status.get("status") != "image_published_and_verified" or not status.get("terminated"):
                    raise ValueError("Build lacks verified completion and cleanup")
                if (
                    binding["original_image"] != task["original_image"]
                    or binding["dsh_image"].split("@", 1)[0] != task["image_tag"].rsplit(":", 1)[0]
                ):
                    raise ValueError("Published image does not match task")
                merged[task["instance_id"]] = {
                    "instance_id": task["instance_id"],
                    "dataset_image": task["docker_image"],
                    "mapped_image": task["mapped_image"],
                    "original_image": binding["original_image"],
                    "dsh_image": binding["dsh_image"],
                }
                result.update(
                    status="passed",
                    dsh_image=binding["dsh_image"],
                    binding_sha256=sha(task_root / "build/image-binding.json"),
                )
        except Exception as exc:
            result.update(status="failed", error_type=type(exc).__name__)
            plan["status"] = "failed"
            return 1
        finally:
            (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    combined = {
        "schema": "xdan.mimo-dsh-image-bindings.v1",
        "bindings": [merged[task["instance_id"]] for task in plan["tasks"]],
    }
    data._bindings(json.dumps(combined).encode(), instances, mapping)
    (args.output / "image-bindings.json").write_text(json.dumps(combined, indent=2) + "\n")
    plan["status"] = "images-ready-data-not-prepared"
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-parquet",
        "image-mapping",
        "existing-bindings",
        "builder-source",
        "source-repo",
        "context",
        "release-dir",
        "output",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--original-images", type=Path)
    result.add_argument("--registry-token-file", type=Path)
    result.add_argument("--registry-user", default="cryptoSUN2049")
    result.add_argument("--registry-prefix", default="ghcr.io/cryptosun2049/mimo-dsh-code")
    result.add_argument("--tag-suffix", required=True)
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--execute-build", action="store_true")
    return result


def main():
    args = parser().parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.absolute())
    plan, *rest = prepare_plan(args)
    code = execute_plan(args, plan, *rest) if args.execute_build else 0
    print(
        json.dumps(
            {
                "status": plan["status"],
                "train_ids": plan["train_ids"],
                "holdout_ids": plan["holdout_ids"],
                "missing_original_digests": plan["missing_original_digests"],
            }
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
