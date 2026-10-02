"""Prepare disjoint, pinned MiMo Code parquet files; never execute task content."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from pathlib import Path

SOURCE_REPOSITORY = "XiaomiMiMo/MiMo-V2.6-RL-oss"
SOURCE_REVISION = "639865fd3374018d6cb29b9fb82dd531406fcf5f"
SOURCE_SHA256 = "e15733cf2451cfbc5492a4120f7f8cfddbad818aa9f0b324c79888dd1fece161"
MAPPING_SHA256 = "704ad2716d746430a805b9b1ac7c6725aea2117fd8e4823ebe9d339d0313ad58"
DEFAULT_TRAIN = "format-code-task-001661"
DEFAULT_HOLDOUT = "format-code-task-001457"
INSTANCE_FIELDS = {
    "dataset_type",
    "docker_image",
    "cwd",
    "instance_id",
    "problem_statement",
    "test_patch",
    "test_command",
    "verifier_timeout_sec",
}
BINDING_FIELDS = {"instance_id", "dataset_image", "mapped_image", "original_image", "dsh_image"}
DIGEST_IMAGE = re.compile(r"[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}\Z")
TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pinned_bytes(path, expected) -> bytes:
    data = Path(path).read_bytes()
    if _sha256(data) != expected:
        raise ValueError(f"SHA256 mismatch for {Path(path).name}")
    return data


def _normal_prompt(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _instances(rows):
    by_id, prompt_ids = {}, {}
    for index, row in enumerate(rows):
        extra = row.get("extra_info")
        if not isinstance(extra, dict) or not isinstance(extra.get("instance_json"), str):
            raise ValueError(f"Row {index} requires extra_info.instance_json string")
        instance = json.loads(extra["instance_json"])
        if not isinstance(instance, dict) or set(instance) != INSTANCE_FIELDS:
            raise ValueError(f"Row {index} must preserve the eight-field Code contract")
        for key in INSTANCE_FIELDS - {"verifier_timeout_sec"}:
            if not isinstance(instance[key], str) or not instance[key].strip() or "\x00" in instance[key]:
                raise ValueError(f"Row {index} has invalid {key}")
        task_id = instance["instance_id"]
        if not TASK_ID.fullmatch(task_id) or instance["dataset_type"] != "opensource-code":
            raise ValueError(f"Row {index} is not a valid Code instance")
        if extra.get("instance_id", task_id) != task_id:
            raise ValueError(f"Row {index} has conflicting instance_id")
        if type(instance["verifier_timeout_sec"]) is not int or instance["verifier_timeout_sec"] <= 0:
            raise ValueError(f"Row {index} has invalid verifier timeout")
        if row.get("prompt") != [{"role": "user", "content": instance["problem_statement"]}]:
            raise ValueError(f"Row {index} prompt must equal its original problem_statement")
        if task_id in by_id:
            raise ValueError(f"Duplicate source instance_id: {task_id}")
        normalized = _normal_prompt(instance["problem_statement"])
        if normalized in prompt_ids:
            raise ValueError(f"Duplicate normalized prompt: {prompt_ids[normalized]} / {task_id}")
        prompt_ids[normalized] = task_id
        by_id[task_id] = (index, instance)
    return by_id


def _mapping(data: bytes):
    mapping = {}
    for line in data.decode().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or set(item) != {"dataset_image", "dockerhub_image"}:
            raise ValueError("Invalid official image mapping")
        source, mapped = item["dataset_image"], item["dockerhub_image"]
        if not isinstance(source, str) or not isinstance(mapped, str) or not source or not mapped:
            raise ValueError("Invalid official image mapping values")
        if source in mapping and mapping[source] != mapped:
            raise ValueError(f"Conflicting official image mapping: {source}")
        mapping[source] = mapped
    return mapping


def _bindings(data: bytes, instances, mapping):
    document = json.loads(data)
    if not isinstance(document, dict) or document.get("schema") != "xdan.mimo-dsh-image-bindings.v1":
        raise ValueError("Unsupported image binding schema")
    entries = document.get("bindings")
    if not isinstance(entries, list):
        raise ValueError("Image bindings must be a list")
    bindings, derived_sources = {}, {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != BINDING_FIELDS:
            raise ValueError("Invalid image binding fields")
        if any(not isinstance(value, str) or not value for value in entry.values()):
            raise ValueError("Invalid image binding values")
        task_id = entry["instance_id"]
        if task_id not in instances:
            raise ValueError(f"Unknown image binding instance_id: {task_id}")
        if task_id in bindings:
            raise ValueError(f"Duplicate image binding instance_id: {task_id}")
        instance = instances[task_id][1]
        source = instance["docker_image"]
        if entry["dataset_image"] != source or entry["mapped_image"] != mapping.get(source):
            raise ValueError(f"Image binding does not match original mapping: {task_id}")
        for key in ("original_image", "dsh_image"):
            if not DIGEST_IMAGE.fullmatch(entry[key]):
                raise ValueError(f"Image binding needs an immutable digest: {task_id} / {key}")
        mapped_repository = entry["mapped_image"].rsplit(":", 1)[0]
        if entry["original_image"].split("@", 1)[0] != mapped_repository:
            raise ValueError(f"Original image repository differs from official mapping: {task_id}")
        derived_digest = entry["dsh_image"].split("@", 1)[1]
        if derived_digest in derived_sources and derived_sources[derived_digest] != source:
            raise ValueError(f"DSH image digest reused across different task images: {task_id}")
        derived_sources[derived_digest] = source
        bindings[task_id] = entry
    return bindings


def prepare_data(*, source_parquet, image_mapping, image_bindings, source_revision, output, train_ids, holdout_ids):
    """Validate all inputs before creating a new output directory; return its manifest."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must be new: {output}")
    if not output.parent.is_dir():
        raise ValueError("Output parent directory must already exist")
    if source_revision != SOURCE_REVISION:
        raise ValueError("Source revision does not match the frozen Code dataset")
    splits = {"train": list(train_ids), "holdout": list(holdout_ids)}
    for split, ids in splits.items():
        if not ids or any(not isinstance(task_id, str) or not TASK_ID.fullmatch(task_id) for task_id in ids):
            raise ValueError(f"{split} must contain valid task IDs")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate task IDs within {split}")
    if set(splits["train"]) & set(splits["holdout"]):
        raise ValueError("Train and holdout instance IDs overlap")
    if DEFAULT_TRAIN in splits["holdout"]:
        raise ValueError("The calibrated 001661 task cannot be holdout")

    source_data = _pinned_bytes(source_parquet, SOURCE_SHA256)
    mapping_data = _pinned_bytes(image_mapping, MAPPING_SHA256)
    binding_data = Path(image_bindings).read_bytes()
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Read exactly the bytes whose digest was checked, not a second filesystem view.
    table = pq.read_table(pa.BufferReader(source_data), use_threads=False)
    rows = table.to_pylist()
    instances = _instances(rows)
    mapping = _mapping(mapping_data)
    bindings = _bindings(binding_data, instances, mapping)
    selected, records = {}, []
    for split, ids in splits.items():
        converted = []
        for task_id in ids:
            if task_id not in instances:
                raise ValueError(f"Unknown selected instance_id: {task_id}")
            if task_id not in bindings:
                raise ValueError(f"Missing image binding: {task_id}")
            index, instance = instances[task_id]
            row = copy.deepcopy(rows[index])
            replacement = {**instance, "docker_image": bindings[task_id]["dsh_image"]}
            original_json = row["extra_info"]["instance_json"]
            row["extra_info"]["instance_json"] = json.dumps(replacement, ensure_ascii=False, separators=(",", ":"))
            converted.append(row)
            records.append(
                {
                    "split": split,
                    "instance_id": task_id,
                    "source_row_index": index,
                    "source_instance_sha256": _sha256(original_json.encode()),
                    "normalized_prompt_sha256": _sha256(_normal_prompt(instance["problem_statement"]).encode()),
                    "image_binding": bindings[task_id],
                }
            )
        selected[split] = pa.Table.from_pylist(converted, schema=table.schema)

    # mkdir(exist_ok=False) is the exclusive boundary; no existing path is overwritten.
    output.mkdir(mode=0o700)
    outputs = {}
    for split, data in selected.items():
        path = output / f"{split}.parquet"
        pq.write_table(data, path)
        path.chmod(0o600)
        outputs[split] = {"file": path.name, "rows": data.num_rows, "sha256": _sha256(path.read_bytes())}
    manifest = {
        "schema": "xdan.mimo-dsh-data.v1",
        "status": "prepared-only",
        "training_ready": False,
        "quality_claim": "none",
        "limitation": "A minimal smoke split, especially one holdout task, cannot establish quality improvement.",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "parquet_sha256": SOURCE_SHA256,
            "mapping_sha256": MAPPING_SHA256,
            "image_bindings_sha256": _sha256(binding_data),
            "rows": table.num_rows,
        },
        "splits": splits,
        "checks": {"unique_source_ids": True, "unique_normalized_prompts": True, "disjoint_splits": True},
        "tasks": records,
        "outputs": outputs,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    manifest_path.chmod(0o600)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--image-mapping", type=Path, required=True)
    parser.add_argument("--image-bindings", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-id", action="append")
    parser.add_argument("--holdout-id", action="append")
    args = parser.parse_args()
    manifest = prepare_data(
        source_parquet=args.source_parquet,
        image_mapping=args.image_mapping,
        image_bindings=args.image_bindings,
        source_revision=args.source_revision,
        output=args.output,
        train_ids=args.train_id if args.train_id is not None else [DEFAULT_TRAIN],
        holdout_ids=args.holdout_id if args.holdout_id is not None else [DEFAULT_HOLDOUT],
    )
    print(json.dumps({"status": manifest["status"], "splits": manifest["splits"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
