"""Official-run deadline interruption and logged-sandbox cleanup; never training acceptance."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

BASE = Path("/workspace/train-p0-dsh-integration")
SOURCE = BASE / "source-a2ad9f61"
SERVICES_SOURCE = BASE / "source-8293685f"
SERVICES_RUN = BASE / "runs/fresh-sft9b-64k-is1-r1-20260930"
SERVICES_MANIFEST_SHA = "55db761fd107f1bc94fba9a2ed3e6bafe2d361984410502d9fe175c6c8334e9d"
BASE_RUN = BASE / "runs/official-code-4gpu-64k-r1-20260930"
PYTHON = Path("/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/venv/bin/python3")
RUNS = {BASE_RUN: "a9off001", BASE_RUN / "resume-step2": "a9off002"}
EXPERIMENTS = {
    BASE_RUN: "official-code-4gpu-64k-r1-20260930-fresh",
    BASE_RUN / "resume-step2": "official-code-4gpu-64k-r1-20260930-resume",
}
BEGIN = datetime.fromisoformat("2026-09-30T12:48:50+00:00").timestamp()
END = datetime.fromisoformat("2026-09-30T12:51:10+00:00").timestamp()
HARD_DEADLINE = datetime.fromisoformat("2026-09-30T12:51:43.743+00:00").timestamp()
PROC = Path("/proc")
RAY_EXECUTABLES = {
    "gcs_server": str(PYTHON.parent.parent / "lib/python3.12/site-packages/ray/core/src/ray/gcs/gcs_server"),
    "raylet": str(PYTHON.parent.parent / "lib/python3.12/site-packages/ray/core/src/ray/raylet/raylet"),
}


def pythonpath(source):
    return ":".join((str(source), str(source / "third_party/mimoagent-osr/src"), str(source / "third_party/uni_agent")))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def window(now=None):
    if not BEGIN <= (time.time() if now is None else now) < END:
        raise RuntimeError("Outside mutation window")


def process(pid):
    fields = None
    try:
        path = PROC / str(pid)
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        command = (path / "cmdline").read_bytes()
        return {
            "pid": pid,
            "ppid": int(fields[1]),
            "start_ticks": int(fields[19]),
            "command_sha256": hashlib.sha256(command).hexdigest(),
            "argv": [s.decode() for s in command.split(b"\0") if s],
            "comm": (path / "comm").read_text().strip(),
            "executable": (
                os.readlink(path / "exe") if (path / "comm").read_text().strip() in {"gcs_server", "raylet"} else None
            ),
        }
    except FileNotFoundError:
        return None
    except PermissionError as error:
        # /proc reads race with exit; permission failure alone is never exit proof.
        try:
            current = (path / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            return None
        if fields is not None and current[19] != fields[19]:
            raise RuntimeError("PID reused during process snapshot") from error
        if fields is not None and current[0] == "Z":
            return None
        raise


def identity(row):
    return {key: row[key] for key in ("pid", "ppid", "start_ticks", "command_sha256")}


def same(left, right):
    return right is not None and all(left[k] == right[k] for k in ("pid", "start_ticks", "command_sha256"))


def env(pid):
    allowed = {b"RUN_DIR", b"WANDB_RUN_ID", b"RAY_JOB_ID", b"PYTHONPATH", b"EXP_NAME"}
    result = {}
    for item in (PROC / str(pid) / "environ").read_bytes().split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep and key in allowed:
            result[key.decode()] = value.decode()
    return result


def signal_owned(row, sig):
    window()
    current = process(row["pid"])
    if current is None:
        return
    if not same(row, current):
        raise RuntimeError("PID identity changed")
    # No os.kill fallback: fail closed on hosts without pidfd.
    try:
        fd = os.pidfd_open(row["pid"])
    except ProcessLookupError:
        return
    try:
        current = process(row["pid"])
        if current is None:
            return
        if not same(row, current):
            raise RuntimeError("PID identity changed after pidfd open")
        signal.pidfd_send_signal(fd, sig)
    finally:
        os.close(fd)


def runs(require_started=True):
    result = []
    targets = []
    for run, wb in RUNS.items():
        receipt = run / "start-receipt.json"
        if not receipt.exists():
            if any((run / name).exists() for name in ("start-claim.json", "train.pid", "train.log")):
                raise RuntimeError("Partial startup without start receipt")
            result.append({"run": str(run), "status": "not_started"})
            continue
        start = read(receipt)
        if start.get("source") != str(SOURCE) or start.get("wandb_run_id") != wb:
            raise RuntimeError("Start source/run mismatch")
        if sha(run / "launch.sh") != start.get("launch_sha256"):
            raise RuntimeError("Launch SHA mismatch")
        pid = int((run / "train.pid").read_text())
        target = process(pid)
        supervisor = process(start["processes"]["train_driver"])
        row = {"run": str(run), "start_receipt_sha256": sha(receipt), "launch_sha256": sha(run / "launch.sh")}
        if target is None:
            if supervisor is not None or not (run / "train-exit.json").is_file():
                raise RuntimeError("Absent driver without completed supervisor")
            exit_record = read(run / "train-exit.json")
            if type(exit_record.get("returncode")) is not int:
                raise RuntimeError("Invalid actual exit receipt")
            row.update(
                status="already_exited",
                train_exit={"returncode": exit_record["returncode"]},
                train_exit_sha256=sha(run / "train-exit.json"),
            )
        else:
            values = env(pid)
            if (
                supervisor is None
                or supervisor["argv"][:2] != [str(PYTHON), str(run / "train_driver.py")]
                or target["ppid"] != supervisor["pid"]
                or values.get("RUN_DIR") != str(run)
                or values.get("WANDB_RUN_ID") != wb
                or (PROC / str(pid) / "cwd").resolve() != SOURCE
            ):
                raise RuntimeError("Driver ownership mismatch")
            row.update(status="active", driver=identity(target), supervisor=identity(supervisor))
            targets.append(target)
        result.append(row)
    if require_started and not any(r["status"] != "not_started" for r in result):
        raise RuntimeError("No actual started run")
    return result, targets


def ray_snapshot():
    records = {}
    for path in PROC.iterdir():
        if path.name.isdecimal():
            row = process(int(path.name))
            if row:
                records[row["pid"]] = row
    heads = [r for r in records.values() if r["comm"] in {"gcs_server", "raylet"}]
    if not heads:
        return [], [], None, []
    if len(heads) != 2 or {r["comm"] for r in heads} != {"gcs_server", "raylet"}:
        raise RuntimeError("Ray head cardinality mismatch")
    sessions = set()
    for row in heads:
        command = " ".join(row["argv"])
        found = set(re.findall(r"/tmp/ray-tp0fresh-r1/session_[A-Za-z0-9_.-]+", command))
        if (
            len(found) != 1
            or row.get("executable") != RAY_EXECUTABLES[row["comm"]]
            or (row["comm"] == "raylet" and str(PYTHON.parent.parent) not in command)
        ):
            raise RuntimeError("Foreign or unattributed Ray head")
        sessions.update(found)
    if len(sessions) != 1 or Path(next(iter(sessions))).is_symlink():
        raise RuntimeError("Ray session mismatch")
    selected = {r["pid"] for r in heads}
    while True:
        added = {p for p, r in records.items() if r["ppid"] in selected} - selected
        if not added:
            break
        selected.update(added)
    jobs = []
    for pid in selected:
        values = env(pid)
        if values.get("RAY_JOB_ID"):
            record = records[pid]
            if (
                record["comm"] == "ray::IDLE"
                and record["argv"] == ["ray::IDLE"]
                and values["RAY_JOB_ID"] == "ffffffff"
                and not any(values.get(k) for k in ("RUN_DIR", "WANDB_RUN_ID", "EXP_NAME"))
                and values.get("PYTHONPATH", "") in {"", pythonpath(SOURCE), pythonpath(SERVICES_SOURCE)}
            ):
                continue
            if values["RAY_JOB_ID"] == "ffffffff":
                raise RuntimeError("Foreign or unattributed Ray job")
            run = next((r for r, wb in RUNS.items() if values.get("WANDB_RUN_ID") == wb), Path("/"))
            if (
                run not in RUNS
                or values.get("EXP_NAME") != EXPERIMENTS[run]
                or ("RUN_DIR" in values and values["RUN_DIR"] != str(run))
                or str(SOURCE) not in values.get("PYTHONPATH", "").split(":")
                or not re.fullmatch(r"[0-9a-f]{8}", values["RAY_JOB_ID"])
            ):
                raise RuntimeError("Foreign or unattributed Ray job")
            jobs.append({"pid": pid, "job_id": values["RAY_JOB_ID"], "run": str(run)})
    if not jobs:
        # Before training, only this new head's precisely identified nil workers may exist.
        # No unassigned worker exception is widened: each nil row passed the strict branch above.
        ready = read(SERVICES_RUN / "services-receipt.json")
        pinned = {r["pid"]: r for r in ready.get("ray_head_processes", [])}
        if (
            ready.get("passed") is not True
            or ready.get("source") != str(SERVICES_SOURCE)
            or ready.get("run_dir") != str(SERVICES_RUN)
            or ready.get("wandb_run_id") != "a9newp01"
            or ready.get("source_manifest_sha256") != SERVICES_MANIFEST_SHA
            or set(pinned) != {r["pid"] for r in heads}
            or not all(same(pinned[r["pid"]], r) for r in heads)
        ):
            raise RuntimeError("No positively attributed Ray job or precisely pinned new idle head")
    return [records[p] for p in sorted(selected)], heads, next(iter(sessions)), jobs


def assert_no_ray_producers():
    if ray_snapshot()[0]:
        raise RuntimeError("New Ray producer appeared")
    # Independently find reparented workers; never kill an unanchored process.
    for path in PROC.iterdir():
        if not path.name.isdecimal():
            continue
        row = process(int(path.name))
        if row and row["comm"].startswith("ray::"):
            raise RuntimeError("Unanchored Ray worker remains; no cleanup claimed")


def merge(original, fresh):
    result = {r["pid"]: r for r in original}
    for row in fresh:
        if row["pid"] in result and not same(result[row["pid"]], row):
            raise RuntimeError("PID reused while capturing descendants")
        result[row["pid"]] = row
    return list(result.values())


def wait_absent(rows, seconds):
    until = min(END, time.time() + seconds)
    while time.time() < until:
        current = [process(row["pid"]) for row in rows]
        if all(r is None for r in current):
            return True
        for old, new in zip(rows, current, strict=True):
            if new is not None and not same(old, new):
                raise RuntimeError("PID reuse during exit verification")
        time.sleep(0.25)
    return False


def stop_producers(output):
    run_rows, targets = runs()
    owned, heads, session, jobs = ray_snapshot()
    write(
        output / "before.json",
        {"runs": run_rows, "ray": [identity(r) for r in owned], "session": session, "jobs": jobs},
    )
    for target in targets:
        signal_owned(target, signal.SIGTERM)
    wait_absent(targets, 8)
    # Re-snapshot while original heads still anchor descendant ownership.
    fresh, new_heads, new_session, new_jobs = ray_snapshot()
    if session != new_session or {r["pid"] for r in heads} != {r["pid"] for r in new_heads}:
        raise RuntimeError("Ray session changed during shutdown")
    owned = merge(owned, fresh)
    write(output / "producer-inventory.json", {"processes": [identity(r) for r in owned], "jobs": new_jobs})
    # Freeze the captured tree before the final snapshots: descendants can themselves fork.
    frozen = set()
    for _ in range(4):
        for row in owned:
            if row["pid"] not in frozen:
                signal_owned(row, signal.SIGSTOP)
                frozen.add(row["pid"])
        fresh, _, final_session, _ = ray_snapshot()
        if final_session != session:
            raise RuntimeError("Ray session changed after freeze")
        owned = merge(owned, fresh)
        if {r["pid"] for r in owned} <= frozen:
            break
    else:
        raise RuntimeError("Producer tree did not stabilize")
    write(output / "frozen-inventory.json", {"processes": [identity(r) for r in owned]})
    for row in owned:
        signal_owned(row, signal.SIGTERM)
    for row in owned:
        signal_owned(row, signal.SIGCONT)
    wait_absent(owned + targets, 5)
    for row in owned + targets:
        signal_owned(row, signal.SIGKILL)
    if not wait_absent(owned + targets, 10):
        raise RuntimeError("Producer still alive")
    # A later session is never silently killed.
    assert_no_ray_producers()
    until = min(END, time.time() + 5)
    while True:
        try:
            after, remaining = runs()
            break
        except RuntimeError:
            if time.time() >= until:
                raise
            time.sleep(0.25)
    if remaining:
        raise RuntimeError("Driver still active")
    write(output / "producers-stopped.json", {"verified": True, "runs": after, "at": time.time()})


def sandbox_ids():
    ids = set()
    files = []
    for run in RUNS:
        path = run / "train.log"
        if not path.exists():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                ids.update(v.decode() for v in re.findall(rb"\bsb-[A-Za-z0-9]+", line))
        files.append({"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
    return ids, files


def cleanup_one(sid, factory):
    window()
    sandbox = factory(sid)
    before = sandbox.poll()
    if before is None:
        sandbox.terminate()
    # New handle: independent poll, not just terminate's return value.
    after = factory(sid).poll()
    while after is None and time.time() < END - 2:
        time.sleep(0.5)
        after = factory(sid).poll()
    if type(after) is not int:
        raise RuntimeError("Sandbox not terminal")
    return {"sandbox_id": sid, "before": before, "after": after}


def cleanup(output, factory):
    previous = None
    for attempt in range(3):
        window()
        ids, files = sandbox_ids()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        try:
            rows = list(pool.map(lambda sid: cleanup_one(sid, factory), sorted(ids)))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        write(output / f"cleanup-{attempt}.json", {"rows": rows, "logs": files})
        time.sleep(1)
        after, _ = sandbox_ids()
        if ids == after == previous:
            return {"sandbox_count": len(ids), "rounds": attempt + 1}
        previous = after
    raise RuntimeError("Sandbox IDs did not stabilize")


def expired(_signum, _frame):
    raise TimeoutError("Cleanup deadline reached")


def hard_deadline():
    time.sleep(max(0, END - time.time()))
    os._exit(124)


def finish(output, status):
    code = 1
    try:
        write(output / "receipt.json", status)
        code = 0 if status.get("status") == "cleanup_verified" else 1
    finally:
        os._exit(code)


def readonly_check():
    run_rows, targets = runs(require_started=False)
    owned, heads, session, jobs = ray_snapshot()
    ids, logs = sandbox_ids()
    return {
        "schema": "xdan.official-deadline-readonly.v1",
        "mutations_started": False,
        "script_sha256": sha(Path(__file__)),
        "runs": run_rows,
        "driver_count": len(targets),
        "ray": [identity(row) for row in owned],
        "heads": [identity(row) for row in heads],
        "session": session,
        "jobs": jobs,
        "sandbox_ids": sorted(ids),
        "logs": logs,
        "mutation_not_before": BEGIN,
        "cleanup_deadline": END,
        "independent_pod_deadline": HARD_DEADLINE,
        "training_acceptance": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        if args.output is not None:
            raise ValueError("--check prints to stdout and never creates output files")
        print(json.dumps(readonly_check(), indent=2, allow_nan=False))
        return
    if args.output is None:
        raise ValueError("--execute requires --output")
    approved = BASE / "operations-official-baseline-20260930/deadline-close"
    if not args.output.resolve().is_relative_to(approved) or args.output.resolve() == approved:
        raise ValueError("Output must be a new child of the fixed operation directory")
    args.output.mkdir(parents=True, exist_ok=False)
    threading.Thread(target=hard_deadline, name="fixed-deadline", daemon=True).start()
    self_row = process(os.getpid())
    if self_row is None:
        raise RuntimeError("Cannot establish own process identity")
    write(
        args.output / "armed.json",
        {
            "schema": "xdan.deadline-close-armed.v1",
            "process": identity(self_row),
            "script_sha256": sha(Path(__file__)),
            "begin": BEGIN,
            "end": END,
            "armed_at": time.time(),
            "mutations_started": False,
        },
    )
    while time.time() < BEGIN:
        time.sleep(min(30, BEGIN - time.time()))
    window()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, END - 2 - time.time()))
    status = {
        "schema": "xdan.deadline-producer-cleanup.v1",
        "training_acceptance": False,
        "script_sha256": sha(Path(__file__)),
        "mutation_not_before": BEGIN,
        "cleanup_deadline": END,
        "limitation": "Only persisted IDs are covered; unlogged created sandboxes remain unproven",
    }
    try:
        stop_producers(args.output)
        os.environ["MODAL_CONFIG_PATH"] = "/root/mimo-private/modal.toml"
        os.environ["MODAL_PROFILE"] = "l98348740"
        import modal

        result = cleanup(args.output, modal.Sandbox.from_id)
        assert_no_ray_producers()
        status.update(status="cleanup_verified", **result)
    except Exception as error:
        status.update(status="incomplete", error_type=type(error).__name__)
    finally:
        finish(args.output, status)


if __name__ == "__main__":
    main()
