"""Offline image orchestration contracts; subprocesses never publish real images."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[3] / "scripts/code/dsh/build_task_images.py"


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("image_builder_test", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def inputs(tmp_path, module, monkeypatch):
    args = module.parser().parse_args(
        [
            "--source-parquet",
            str(tmp_path / "source.parquet"),
            "--image-mapping",
            str(tmp_path / "mapping.jsonl"),
            "--existing-bindings",
            str(tmp_path / "existing.json"),
            "--builder-source",
            str(tmp_path / "builder"),
            "--source-repo",
            str(tmp_path / "repo"),
            "--context",
            str(tmp_path / "context"),
            "--release-dir",
            str(tmp_path / "release"),
            "--output",
            str(tmp_path / "output"),
            "--tag-suffix",
            "fixture-r1",
        ]
    )
    for root in (args.builder_source, args.context, args.release_dir, args.source_repo):
        root.mkdir()
    tools = {}
    for name in module.TOOL_HASHES:
        path = args.builder_source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
        tools[name] = module.sha(path)
    monkeypatch.setattr(module, "TOOL_HASHES", tools)
    lock = {"wheels": {"fake.whl": hashlib.sha256(b"wheel").hexdigest()}}
    (args.release_dir / "fake.whl").write_bytes(b"wheel")
    (args.context / "python.tar.gz").write_bytes(b"python")
    (args.context / "build-inputs.json").write_text(
        json.dumps({"lock": lock, "source_commit": "a" * 40, "source_files": {}})
    )
    files = {name: module.sha(args.context / name) for name in ("python.tar.gz", "build-inputs.json")}
    (args.context / "context-manifest.json").write_text(json.dumps({"files": files, "source_commit": "a" * 40}))
    args.source_parquet.write_bytes(b"source")
    args.image_mapping.write_bytes(b"mapping")
    ids = [module.TRAIN, module.HOLDOUT, *[f"format-code-task-{index:06d}" for index in range(10)]]
    tasks = {
        key: {
            "instance_id": key,
            "docker_image": key + ":latest",
            "mapped_image": "docker.io/original/tasks:" + key,
            "cwd": "/work " + key,
            "dataset_type": "opensource-code",
        }
        for key in ids
    }
    existing = {"schema": "xdan.mimo-dsh-image-bindings.v1", "bindings": []}
    for index, key in enumerate((module.TRAIN, module.HOLDOUT)):
        existing["bindings"].append(
            {
                "instance_id": key,
                "dataset_image": tasks[key]["docker_image"],
                "mapped_image": tasks[key]["mapped_image"],
                "original_image": "docker.io/original/tasks@sha256:" + str(index) * 64,
                "dsh_image": "ghcr.io/fixture/image@sha256:" + str(index) * 64,
            }
        )
    args.existing_bindings.write_text(json.dumps(existing))
    monkeypatch.setattr(module, "read_tasks", lambda _: (module.data_module(), tasks))
    return args, tasks


def provide_originals(args, tasks, module):
    args.original_images = args.output.parent / "originals.json"
    args.original_images.write_text(
        json.dumps(
            {
                key: "docker.io/original/tasks@sha256:" + hashlib.sha256(key.encode()).hexdigest()
                for key in tasks
                if key not in (module.TRAIN, module.HOLDOUT)
            }
        )
    )


def test_selection_is_order_independent_and_disjoint(module):
    ids = [module.TRAIN, module.HOLDOUT, *[f"task-{i}" for i in range(20)]]
    result = module.select_tasks(ids)
    assert result == module.select_tasks(reversed(ids))
    assert len(result) == len(set(result)) == 8
    assert result[0] == module.TRAIN and module.HOLDOUT not in result
    with pytest.raises(ValueError):
        module.select_tasks(ids + [ids[0]])
    with pytest.raises(ValueError):
        module.select_tasks([module.TRAIN, module.HOLDOUT])


def test_real_parquet_extracts_only_public_metadata(tmp_path, module, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    args = SimpleNamespace(source_parquet=tmp_path / "source.parquet", image_mapping=tmp_path / "mapping.jsonl")
    instance = {
        "instance_id": module.TRAIN,
        "docker_image": "task:latest",
        "cwd": "/workspace/repo",
        "dataset_type": "opensource-code",
        "test_patch": "opaque synthetic fixture never returned",
        "test_command": "do-not-run",
    }
    pq.write_table(pa.Table.from_pylist([{"extra_info": {"instance_json": json.dumps(instance)}}]), args.source_parquet)
    args.image_mapping.write_text(json.dumps({"dataset_image": "task:latest", "dockerhub_image": "docker.io/a/b:task"}))
    data = module.data_module()
    data.SOURCE_SHA256 = module.sha(args.source_parquet)
    data.MAPPING_SHA256 = module.sha(args.image_mapping)
    monkeypatch.setattr(module, "data_module", lambda: data)
    _, tasks = module.read_tasks(args)
    assert set(tasks[module.TRAIN]) == {"instance_id", "docker_image", "cwd", "dataset_type", "mapped_image"}
    assert "opaque" not in json.dumps(tasks)
    args.source_parquet.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.read_tasks(args)


def test_default_plan_no_remote_calls(inputs, module, monkeypatch):
    args, _ = inputs
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Must not execute a builder"))
    plan, *_ = module.prepare_plan(args)
    assert len(plan["missing_original_digests"]) == 7
    assert plan["holdout_ids"] == [module.HOLDOUT]
    assert len(plan["sha256"]["builder_tools"]) == 6
    assert plan["status"] == "planned"
    with pytest.raises(FileExistsError):
        module.prepare_plan(args)


def test_expanded_cli_and_task_paths(inputs, module):
    args, tasks = inputs
    provide_originals(args, tasks, module)
    plan, *_ = module.prepare_plan(args)
    builds = [task for task in plan["tasks"] if not task["reuse"]]
    assert len(builds) == 7
    for task in builds:
        commands = task["commands"]
        assert commands["probe"][1:4] == ["-m", "deployment.harbor.mimo.remote_builder", "probe"]
        for stage in ("probe", "build"):
            command = commands[stage]
            assert command[command.index("--task-cwd") + 1] == task["cwd"]
            assert command[command.index("--output") + 1] == str(args.output / task["instance_id"] / stage)
        context = commands["context"]
        assert context[context.index("--artifact-dir") + 1] == str(args.release_dir)
        assert context[context.index("--lock") + 1] == str(args.output / "release-lock.json")
    assert len({task["image_tag"] for task in builds}) == 7


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "wrong_repository",
        "existing_changed",
        "tool_changed",
        "context_changed",
        "traversal",
        "symlink",
        "suffix",
    ],
)
def test_fail_closed_before_output(inputs, module, fault):
    args, tasks = inputs
    provide_originals(args, tasks, module)
    if fault == "missing":
        args.original_images = None
        args.execute_build = True
    elif fault in ("wrong_repository", "existing_changed"):
        originals = json.loads(args.original_images.read_text())
        key = module.select_tasks(tasks)[1] if fault == "wrong_repository" else module.TRAIN
        originals[key] = (
            ("docker.io/wrong/repo" if fault == "wrong_repository" else "docker.io/original/tasks")
            + "@sha256:"
            + "f" * 64
        )
        args.original_images.write_text(json.dumps(originals))
    elif fault == "tool_changed":
        (args.builder_source / next(iter(module.TOOL_HASHES))).write_text("changed")
    elif fault == "context_changed":
        (args.context / "python.tar.gz").write_text("changed")
    elif fault == "traversal":
        path = args.context / "context-manifest.json"
        document = json.loads(path.read_text())
        document["files"]["../source.parquet"] = module.sha(args.source_parquet)
        path.write_text(json.dumps(document))
    elif fault == "symlink":
        args.output.symlink_to(args.output.parent / "absent")
    else:
        args.tag_suffix = "../../escape"
    with pytest.raises((ValueError, FileExistsError)):
        module.prepare_plan(args)
    assert not args.output.exists()


def test_execute_requires_credential_path_without_reading_it(inputs, module):
    args, tasks = inputs
    provide_originals(args, tasks, module)
    args.execute_build = True
    with pytest.raises(ValueError, match="credential file path"):
        module.prepare_plan(args)


def test_failure_records_returncode_and_preserves_output(inputs, module, monkeypatch):
    args, tasks = inputs
    provide_originals(args, tasks, module)
    args.registry_token_file = args.output.parent / "never-read-private-path"
    args.execute_build = True
    plan, *rest = module.prepare_plan(args)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == args.builder_source
        assert kwargs["env"]["PYTHONPATH"] == str(args.builder_source)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.execute_plan(args, plan, *rest) == 1
    assert len(calls) == 1
    result = json.loads((args.output / "plan.json").read_text())
    assert result["status"] == "failed"
    assert result["results"][-1]["stages"] == [{"stage": "probe", "returncode": 7}]
    assert not (args.output / "image-bindings.json").exists()
    with pytest.raises(FileExistsError):
        module.prepare_plan(args)


@pytest.mark.parametrize("wrong_image", [False, True])
def test_mock_build_success_and_wrong_task_rejection(inputs, module, monkeypatch, wrong_image):
    args, tasks = inputs
    provide_originals(args, tasks, module)
    args.registry_token_file = args.output.parent / "never-read-private-path"
    args.execute_build = True
    plan, *rest = module.prepare_plan(args)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        if command[2] == "deployment.harbor.mimo.remote_builder" and command[3] == "build":
            task = next(task for task in plan["tasks"] if task["instance_id"] == output.parent.name)
            repository = "ghcr.io/wrong/task" if wrong_image else task["image_tag"].rsplit(":", 1)[0]
            binding = {
                "original_image": task["original_image"],
                "dsh_image": repository + "@sha256:" + hashlib.sha256(task["instance_id"].encode()).hexdigest(),
            }
            (output / "image-binding.json").write_text(json.dumps(binding))
            (output / "status.json").write_text(
                json.dumps({"status": "image_published_and_verified", "terminated": True})
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.execute_plan(args, plan, *rest) == int(wrong_image)
    if wrong_image:
        assert len(calls) == 3
        assert not (args.output / "image-bindings.json").exists()
    else:
        assert len(calls) == 21
        merged = json.loads((args.output / "image-bindings.json").read_text())
        assert len(merged["bindings"]) == 9
        assert len({entry["dsh_image"] for entry in merged["bindings"]}) == 9
        previous = json.loads(args.existing_bindings.read_text())
        assert [entry for entry in merged["bindings"] if entry["instance_id"] == module.HOLDOUT] == [
            previous["bindings"][1]
        ]
