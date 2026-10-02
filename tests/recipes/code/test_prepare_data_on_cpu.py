"""Data-only contracts: no task command, model, registry or hidden test executes."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / "scripts/code/dsh/prepare_data.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("prepare_data_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    bindings = []
    mappings = []
    for index, name in enumerate(("task-train", "task-holdout", "task-unused")):
        original = f"task-image-{index}:latest"
        mapped = f"docker.io/official/tasks:task-{index}"
        instance = {
            "dataset_type": "opensource-code",
            "instance_id": name,
            "docker_image": original,
            "cwd": "/testbed" if index else "/workspace/repo",
            "problem_statement": f"Public task {index}",
            "test_patch": f"opaque fixture patch {index}; $(touch should-never-execute)",
            "test_command": "do-not-execute-this-command",
            "verifier_timeout_sec": 1800,
        }
        rows.append(
            {
                "prompt": [{"role": "user", "content": instance["problem_statement"]}],
                "extra_info": {"instance_json": json.dumps(instance), "instance_id": name, "index": index},
                "data_source": "mimoagent/swe",
                "reward_model": {"style": "rule", "ground_truth": ""},
            }
        )
        mappings.append({"dataset_image": original, "dockerhub_image": mapped})
        bindings.append(
            {
                "instance_id": name,
                "dataset_image": original,
                "mapped_image": mapped,
                "original_image": "docker.io/official/tasks@sha256:" + str(index + 1) * 64,
                "dsh_image": f"ghcr.io/our/tasks-{index}@sha256:" + str(index + 4) * 64,
            }
        )
    source = tmp_path / "code.parquet"
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b"source-note": b"keep-me"})
    pq.write_table(table, source)
    mapping = tmp_path / "image-mapping.jsonl"
    mapping.write_text("".join(json.dumps(entry) + "\n" for entry in mappings))
    binding_path = tmp_path / "bindings.json"
    binding_path.write_text(json.dumps({"schema": "xdan.mimo-dsh-image-bindings.v1", "bindings": bindings}))
    monkeypatch.setattr(module, "SOURCE_SHA256", digest(source))
    monkeypatch.setattr(module, "MAPPING_SHA256", digest(mapping))
    kwargs = dict(
        source_parquet=source,
        image_mapping=mapping,
        image_bindings=binding_path,
        source_revision=module.SOURCE_REVISION,
        output=tmp_path / "prepared",
        train_ids=["task-train"],
        holdout_ids=["task-holdout"],
    )
    return module, kwargs, rows, bindings


def test_roundtrip_changes_only_task_image_and_records_provenance(prepared_inputs):
    module, kwargs, rows, bindings = prepared_inputs
    manifest = module.prepare_data(**kwargs)
    output = kwargs["output"]
    assert set(p.name for p in output.iterdir()) == {"train.parquet", "holdout.parquet", "manifest.json"}
    for index, split in enumerate(("train", "holdout")):
        table = pq.read_table(output / f"{split}.parquet")
        assert table.schema.equals(pq.read_schema(kwargs["source_parquet"]), check_metadata=True)
        actual = table.to_pylist()[0]
        original = copy.deepcopy(rows[index])
        instance = json.loads(actual["extra_info"]["instance_json"])
        assert instance.pop("docker_image") == bindings[index]["dsh_image"]
        source_instance = json.loads(original["extra_info"].pop("instance_json"))
        source_instance.pop("docker_image")
        assert instance == source_instance
        actual["extra_info"].pop("instance_json")
        assert actual == original
        assert manifest["outputs"][split]["sha256"] == digest(output / f"{split}.parquet")
    assert manifest["status"] == "prepared-only"
    assert manifest["training_ready"] is False
    assert manifest["quality_claim"] == "none"
    assert manifest["splits"] == {"train": ["task-train"], "holdout": ["task-holdout"]}
    assert json.loads((output / "manifest.json").read_text()) == manifest
    text = (output / "manifest.json").read_text()
    assert "opaque fixture patch" not in text
    assert "do-not-execute" not in text
    assert "Public task" not in text


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_revision", "0" * 40),
        ("train_ids", []),
        ("holdout_ids", []),
        ("train_ids", ["task-train", "task-train"]),
        ("holdout_ids", ["task-train"]),
        ("holdout_ids", ["unknown-task"]),
    ],
)
def test_invalid_split_or_revision_fails_before_output(prepared_inputs, field, value):
    module, kwargs, _, _ = prepared_inputs
    kwargs[field] = value
    with pytest.raises(ValueError):
        module.prepare_data(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("pin", ["SOURCE_SHA256", "MAPPING_SHA256"])
def test_source_and_mapping_hashes_are_mandatory(prepared_inputs, monkeypatch, pin):
    module, kwargs, _, _ = prepared_inputs
    monkeypatch.setattr(module, pin, "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        module.prepare_data(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("collision", ["id", "prompt", "prompt-normalized"])
def test_source_duplicate_id_or_prompt_is_rejected(prepared_inputs, monkeypatch, collision):
    module, kwargs, rows, _ = prepared_inputs
    instance = json.loads(rows[1]["extra_info"]["instance_json"])
    if collision == "id":
        instance["instance_id"] = "task-train"
        rows[1]["extra_info"]["instance_id"] = "task-train"
    else:
        statement = "Public task 0" if collision == "prompt" else "  ＰＵＢＬＩＣ   TASK 0  "
        instance["problem_statement"] = statement
        rows[1]["prompt"][0]["content"] = statement
    rows[1]["extra_info"]["instance_json"] = json.dumps(instance)
    pq.write_table(pa.Table.from_pylist(rows), kwargs["source_parquet"])
    monkeypatch.setattr(module, "SOURCE_SHA256", digest(kwargs["source_parquet"]))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        module.prepare_data(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.parametrize(
    "fault", ["missing", "tag", "wrong-source", "wrong-map", "wrong-registry", "duplicate", "reuse", "unknown"]
)
def test_invalid_image_binding_fails_before_output(prepared_inputs, fault):
    module, kwargs, _, bindings = prepared_inputs
    if fault == "missing":
        bindings.pop(1)
    elif fault == "tag":
        bindings[1]["dsh_image"] = "ghcr.io/our/task:latest"
    elif fault == "wrong-source":
        bindings[1]["dataset_image"] = bindings[0]["dataset_image"]
    elif fault == "wrong-map":
        bindings[1]["mapped_image"] = bindings[0]["mapped_image"]
    elif fault == "wrong-registry":
        bindings[1]["original_image"] = "ghcr.io/unrelated/tasks@sha256:" + "a" * 64
    elif fault == "duplicate":
        bindings.append(bindings[0])
    elif fault == "reuse":
        bindings[1]["dsh_image"] = bindings[0]["dsh_image"]
    elif fault == "unknown":
        bindings[1]["instance_id"] = "unknown-task"
    kwargs["image_bindings"].write_text(json.dumps({"schema": "xdan.mimo-dsh-image-bindings.v1", "bindings": bindings}))
    with pytest.raises(ValueError):
        module.prepare_data(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("existing", ["directory", "file", "symlink"])
def test_existing_output_is_never_overwritten(prepared_inputs, existing):
    module, kwargs, _, _ = prepared_inputs
    output = kwargs["output"]
    if existing == "directory":
        output.mkdir()
        (output / "sentinel").write_text("keep")
    elif existing == "file":
        output.write_text("keep")
    else:
        output.symlink_to(output.parent / "absent-target")
    with pytest.raises(FileExistsError):
        module.prepare_data(**kwargs)
    if existing == "directory":
        assert (output / "sentinel").read_text() == "keep"
    elif existing == "file":
        assert output.read_text() == "keep"
    else:
        assert output.is_symlink()


def test_calibrated_task_cannot_be_used_as_holdout(prepared_inputs):
    module, kwargs, _, _ = prepared_inputs
    kwargs["holdout_ids"] = [module.DEFAULT_TRAIN]
    with pytest.raises(ValueError, match="calibrated"):
        module.prepare_data(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("fault", ["missing-field", "extra-field", "wrong-prompt", "wrong-id", "wrong-timeout"])
def test_original_contract_is_not_silently_repaired(prepared_inputs, monkeypatch, fault):
    module, kwargs, rows, _ = prepared_inputs
    instance = json.loads(rows[0]["extra_info"]["instance_json"])
    if fault == "missing-field":
        instance.pop("test_command")
    elif fault == "extra-field":
        instance["unexpected"] = "value"
    elif fault == "wrong-prompt":
        rows[0]["prompt"][0]["content"] = "silently replaced prompt"
    elif fault == "wrong-id":
        rows[0]["extra_info"]["instance_id"] = "different-id"
    elif fault == "wrong-timeout":
        instance["verifier_timeout_sec"] = True
    rows[0]["extra_info"]["instance_json"] = json.dumps(instance)
    pq.write_table(pa.Table.from_pylist(rows), kwargs["source_parquet"])
    monkeypatch.setattr(module, "SOURCE_SHA256", digest(kwargs["source_parquet"]))
    with pytest.raises(ValueError):
        module.prepare_data(**kwargs)
    assert not kwargs["output"].exists()


def test_cli_uses_the_same_validated_contract(prepared_inputs, monkeypatch, capsys):
    module, kwargs, _, _ = prepared_inputs
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--source-parquet",
            str(kwargs["source_parquet"]),
            "--image-mapping",
            str(kwargs["image_mapping"]),
            "--image-bindings",
            str(kwargs["image_bindings"]),
            "--source-revision",
            kwargs["source_revision"],
            "--output",
            str(kwargs["output"]),
            "--train-id",
            "task-train",
            "--holdout-id",
            "task-holdout",
        ],
    )
    module.main()
    message = json.loads(capsys.readouterr().out)
    assert message["status"] == "prepared-only"
    assert message["splits"] == {"train": ["task-train"], "holdout": ["task-holdout"]}
    assert (kwargs["output"] / "manifest.json").is_file()
