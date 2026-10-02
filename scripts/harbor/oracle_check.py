# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Policy-free check of Harbor rows on the real sandbox backend.

For each row: start the task environment exactly as training would, grade the untouched task
(expected 0 with a valid verdict), then run the task's reference solution/solve.sh and grade again
(expected 1). Proves image pull, network access for test.sh, and the verifier contract end to end.

usage: oracle_check.py --data train.parquet --harness config/agent/harbor/mini-mimocode-modal.yaml
       [--tasks-root DIR] [--out report.json]
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
            environment.env.copy_to(str(solution), "/oracle/solve.sh")
            run = environment.execute("bash /oracle/solve.sh", cwd=environment.repo_path, timeout=900)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--tasks-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import pandas as pd
    import yaml

    environment_config = dict(yaml.safe_load(args.harness.read_text())["environment"])
    reports = []
    for row in pd.read_parquet(args.data).to_dict("records"):
        instance = json.loads(row["extra_info"]["instance_json"])
        solution = None
        if args.tasks_root is not None:
            candidate = args.tasks_root / instance["instance_id"] / "solution" / "solve.sh"
            solution = candidate if candidate.is_file() else None
        report = _check(instance, environment_config, solution)
        print(json.dumps(report), flush=True)
        reports.append(report)
    ok = all(
        r.get("untouched", {}).get("error_category") is None
        and r.get("solved", {}).get("reward") == 1.0
        and "exception" not in r
        for r in reports
    )
    summary = {"passed": ok, "tasks": reports, "modal_profile": os.getenv("MODAL_PROFILE")}
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"passed": ok, "n": len(reports)}))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
