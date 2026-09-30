"""CPU ownership and no-mutation checks for the fixed official deadline closer."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("official_close", HERE / "close_at_deadline.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def row(pid, parent, comm, argv, executable=None):
    return dict(
        pid=pid,
        ppid=parent,
        start_ticks=pid * 10,
        command_sha256=f"cmd-{pid}",
        comm=comm,
        argv=argv,
        executable=executable,
    )


@pytest.fixture
def ray(monkeypatch, tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    session = "/tmp/ray-tp0fresh-r1/session_2026_09_30"
    records = {
        10: row(
            10,
            1,
            "gcs_server",
            [m.RAY_EXECUTABLES["gcs_server"], f"--session-dir={session}"],
            m.RAY_EXECUTABLES["gcs_server"],
        ),
        11: row(
            11,
            1,
            "raylet",
            [m.RAY_EXECUTABLES["raylet"], f"--session-dir={session}", str(m.PYTHON.parent.parent)],
            m.RAY_EXECUTABLES["raylet"],
        ),
        12: row(12, 11, "ray::IDLE", ["ray::IDLE"]),
    }
    values = {10: {}, 11: {}, 12: {"RAY_JOB_ID": "ffffffff", "PYTHONPATH": m.pythonpath(m.SERVICES_SOURCE)}}
    ready = {
        "passed": True,
        "source": str(m.SERVICES_SOURCE),
        "run_dir": str(m.SERVICES_RUN),
        "wandb_run_id": "a9newp01",
        "source_manifest_sha256": m.SERVICES_MANIFEST_SHA,
        "ray_head_processes": [m.identity(records[10]), m.identity(records[11])],
    }
    for pid in records:
        (proc / str(pid)).mkdir()
    monkeypatch.setattr(m, "PROC", proc)
    monkeypatch.setattr(m, "process", lambda pid: copy.deepcopy(records.get(pid)))
    monkeypatch.setattr(m, "env", lambda pid: copy.deepcopy(values[pid]))
    monkeypatch.setattr(m, "read", lambda path: ready if path == m.SERVICES_RUN / "services-receipt.json" else None)
    return records, values, ready


def test_precisely_pinned_old_idle_head_is_readable_before_official_start(ray):
    owned, heads, session, jobs = m.ray_snapshot()
    assert {r["pid"] for r in owned} == {10, 11, 12}
    assert {r["pid"] for r in heads} == {10, 11}
    assert session.endswith("session_2026_09_30") and jobs == []


@pytest.mark.parametrize("change", ["argv", "job", "run", "path", "manifest", "head"])
def test_idle_alias_does_not_weaken_other_identity_checks(ray, change):
    records, values, ready = ray
    if change == "argv":
        records[12]["argv"] = ["ray::IDLE", "unexpected"]
    if change == "job":
        values[12]["RAY_JOB_ID"] = "12345678"
    if change == "run":
        values[12]["RUN_DIR"] = "/foreign"
    if change == "path":
        values[12]["PYTHONPATH"] += ":/foreign"
    if change == "manifest":
        ready["source_manifest_sha256"] = "0" * 64
    if change == "head":
        ready["ray_head_processes"][0]["start_ticks"] += 1
    with pytest.raises(RuntimeError):
        m.ray_snapshot()


@pytest.mark.parametrize("phase", ["fresh", "resume"])
def test_official_job_requires_its_exact_phase_experiment(ray, phase):
    records, values, _ = ray
    run = m.BASE_RUN if phase == "fresh" else m.BASE_RUN / "resume-step2"
    records[12].update(comm="ray::TaskRunner", argv=["ray::TaskRunner"])
    values[12] = {
        "RAY_JOB_ID": "12345678",
        "WANDB_RUN_ID": m.RUNS[run],
        "RUN_DIR": str(run),
        "EXP_NAME": m.EXPERIMENTS[run],
        "PYTHONPATH": m.pythonpath(m.SOURCE),
    }
    assert m.ray_snapshot()[3] == [{"pid": 12, "job_id": "12345678", "run": str(run)}]
    values[12]["EXP_NAME"] = "dsh-" + run.name
    with pytest.raises(RuntimeError):
        m.ray_snapshot()


def test_no_start_is_allowed_for_readonly_check_but_partial_start_is_not(monkeypatch, tmp_path):
    run = tmp_path / "run"
    monkeypatch.setattr(m, "RUNS", {run: "a9off001"})
    rows, targets = m.runs(require_started=False)
    assert rows == [{"run": str(run), "status": "not_started"}] and targets == []
    with pytest.raises(RuntimeError, match="No actual started run"):
        m.runs()
    run.mkdir()
    (run / "train.pid").write_text("123")
    with pytest.raises(RuntimeError, match="Partial startup"):
        m.runs(require_started=False)


def test_check_calls_no_mutating_functions(monkeypatch, capsys, ray):
    monkeypatch.setattr(m, "runs", lambda require_started: ([{"status": "not_started"}], []))
    monkeypatch.setattr(m, "sandbox_ids", lambda: ({"sb-known"}, []))

    def forbidden(*args, **kwargs):
        raise AssertionError("mutation was called")

    for name in ("write", "signal_owned", "stop_producers", "cleanup", "window"):
        monkeypatch.setattr(m, name, forbidden)
    monkeypatch.setattr(m, "sha", lambda path: "script-sha")
    m.main(["--check"])
    result = json.loads(capsys.readouterr().out)
    assert result["mutations_started"] is False
    assert result["sandbox_ids"] == ["sb-known"]
    assert result["jobs"] == []


def test_cleanup_only_reads_two_official_run_logs(monkeypatch, tmp_path):
    fresh, resume, old = [tmp_path / name for name in ("fresh", "resume", "old")]
    for path, sid in ((fresh, "sb-fresh"), (resume, "sb-resume"), (old, "sb-foreign")):
        path.mkdir()
        (path / "train.log").write_text(f"sandbox {sid}\n")
    monkeypatch.setattr(m, "RUNS", {fresh: "a9off001", resume: "a9off002"})
    assert m.sandbox_ids()[0] == {"sb-fresh", "sb-resume"}


def test_mutation_window_is_fixed_and_outside_signal_is_rejected(monkeypatch):
    monkeypatch.setattr(m.time, "time", lambda: m.BEGIN - 1)
    monkeypatch.setattr(m, "process", lambda pid: pytest.fail("process read must not precede window gate"))
    with pytest.raises(RuntimeError, match="Outside mutation window"):
        m.signal_owned({"pid": 1}, 15)
    assert m.END - m.BEGIN == 140
    assert m.HARD_DEADLINE - m.END == pytest.approx(33.743)
