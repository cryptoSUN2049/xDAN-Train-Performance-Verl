"""Supervise one official invocation and retain its actual exit status."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/workspace/train-p0-dsh-integration")
SOURCE = BASE / "source-a2ad9f61"
FIRST = BASE / "runs/official-code-4gpu-64k-r1-20260930"
DEADLINE = datetime.fromisoformat("2026-09-30T12:51:43.743+00:00")


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fresh", "resume"), required=True)
    args = parser.parse_args()
    run = FIRST if args.phase == "fresh" else FIRST / "resume-step2"
    wb = "a9off001" if args.phase == "fresh" else "a9off002"
    if datetime.now(timezone.utc) >= DEADLINE:
        raise RuntimeError("Fixed GPU deadline reached")
    launcher = run / "launch.sh"
    claim = {"phase": args.phase, "source": str(SOURCE), "wandb_run_id": wb}
    write(run / "start-claim.json", claim)
    with (run / "train.log").open("x") as log:
        child = subprocess.Popen(
            ["bash", str(launcher), args.phase],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=SOURCE,
        )
        with (run / "train.pid").open("x") as stream:
            stream.write(str(child.pid))
            stream.flush()
            os.fsync(stream.fileno())
        write(
            run / "start-receipt.json",
            {
                **claim,
                "launch_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
                "processes": {"train_driver": os.getpid()},
                "child_pid": child.pid,
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        code = child.wait()
    write(
        run / "train-exit.json",
        {
            "returncode": code,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
