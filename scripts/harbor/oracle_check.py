# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Policy-free check of Harbor rows on the real sandbox backend.

For each row: start the task environment exactly as training would, grade the untouched task
(expected 0 with a valid verdict), then upload the task's solution/ to /solution, run solve.sh and grade again
(expected 1). Proves image pull, network access for test.sh, and the verifier contract end to end.

usage: oracle_check.py --data train.parquet --harness config/agent/harbor/mini-mimocode-modal.yaml
       [--tasks-root DIR] [--out report.json] [--jsonl results.jsonl] [--workers 16]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _check(instance: dict, environment_config: dict, solution: Path | None) -> dict:
    from mimoagent.environments.utils import make_dataset_env

    from recipes.harbor.environment import register

    register()
    report: dict = {"instance_id": instance["instance_id"], "image": instance["docker_image"]}
    started = time.time()
    environment = make_dataset_env(instance, **environment_config)
    try:
        environment.setup_environment()
        reward, _output, extra = environment.calculate_reward()
        report["untouched"] = {"reward": reward, "error_category": extra.get("error_category")}
        if solution is not None:
            # Harbor convention: the whole solution/ directory is mounted at /solution.
            environment.env.copy_to(str(solution), "/solution")
            run = environment.execute("bash /solution/solve.sh", cwd=environment.repo_path, timeout=900)
            report["solve_returncode"] = run.get("returncode")
            reward, output, extra = environment.calculate_reward()
            report["solved"] = {"reward": reward, "error_category": extra.get("error_category")}
            if extra.get("error_category") or reward != 1.0:
                report["solved"]["tail"] = output[-1500:]
    except Exception as error:  # noqa: BLE001 - reported, the check never trains
        report["exception"] = repr(error)[:2000]
    finally:
        environment.cleanup()
    report["seconds"] = round(time.time() - started, 1)
    return report


def audit_passed(report: dict) -> bool:
    """nop=0 / oracle=1: the untouched task fails with a valid verdict and the reference solution passes."""
    untouched, solved = report.get("untouched", {}), report.get("solved", {})
    return (
        "exception" not in report
        and untouched.get("error_category") is None
        and untouched.get("reward") == 0.0
        and solved.get("error_category") is None
        and solved.get("reward") == 1.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path)
    parser.add_argument("--out", type=Path, help="summary JSON")
    parser.add_argument("--jsonl", type=Path, help="per-task results, appended as they finish; reruns skip done tasks")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import pandas as pd
    import yaml

    environment_config = dict(yaml.safe_load(args.harness.read_text())["environment"])
    done: dict[str, dict] = {}
    if args.jsonl and args.jsonl.exists():
        for line in args.jsonl.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["instance_id"]] = record
    instances = [
        json.loads(row["extra_info"]["instance_json"]) for row in pd.read_parquet(args.data).to_dict("records")
    ]
    todo = [instance for instance in instances if instance["instance_id"] not in done]
    lock = threading.Lock()

    def work(instance: dict) -> dict:
        solution = None
        if args.tasks_root is not None:
            candidate = args.tasks_root / instance["instance_id"] / "solution"
            solution = candidate if (candidate / "solve.sh").is_file() else None
        report = _check(instance, environment_config, solution)
        report["audit_passed"] = audit_passed(report)
        with lock:
            print(json.dumps(report), flush=True)
            if args.jsonl:
                with args.jsonl.open("a") as stream:
                    stream.write(json.dumps(report) + "\n")
        return report

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for future in as_completed([pool.submit(work, instance) for instance in todo]):
            report = future.result()
            done[report["instance_id"]] = report
    reports = [done[instance["instance_id"]] for instance in instances]
    passed = [r["instance_id"] for r in reports if audit_passed(r)]
    summary = {
        "passed": len(passed) == len(reports),
        "n": len(reports),
        "n_passed": len(passed),
        "passed_ids": passed,
        "tasks": reports,
        "modal_profile": os.getenv("MODAL_PROFILE"),
    }
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"passed": summary["passed"], "n": len(reports), "n_passed": len(passed)}))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
