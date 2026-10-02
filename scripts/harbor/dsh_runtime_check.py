# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Model-free check that the DSH payload injects and boots inside each Harbor task image.

For each row: start the task environment as training would, inject /opt/dsh through DshSdkAgent's
payload path, then run the runtime's keyless smoke (identity hashes + SDK boot, no model calls).

usage: dsh_runtime_check.py --data train.parquet --harness config/agent/harbor/dsh-sdk-modal.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace


def _check(instance: dict, config: dict) -> dict:
    from mimoagent.environments.utils import make_dataset_env

    from recipes.code.dsh_agent import DshSdkAgent
    from recipes.harbor.environment import register

    register()
    agent_config = {k: v for k, v in config["agent"].items() if k != "type"}
    report: dict = {"instance_id": instance["instance_id"], "image": instance["docker_image"]}
    started = time.time()
    environment_config = dict(config["environment"])
    # Same environment construction as recipes/code/mimoagent_runner.py, including the runtime git strip.
    if instance.get("dataset_type") == "opensource-code" and environment_config.get("git_leak_prevention") == "strip":
        from recipes.code.code_environment import make_code_dataset_env

        environment = make_code_dataset_env(instance, **environment_config)
    else:
        environment = make_dataset_env(instance, **environment_config)
    try:
        environment.setup_environment()
        model = SimpleNamespace(config=SimpleNamespace(model_name="policy", model_kwargs={}))
        agent = DshSdkAgent(model, environment.env, **agent_config)
        agent._ensure_runtime()
        smoke = environment.execute(
            "DSH_RUNTIME_MODE=exe /opt/dsh/bin/python /opt/dsh/checks/smoke.py", cwd="/", timeout=300
        )
        report["smoke_returncode"] = smoke.get("returncode")
        output = str(smoke.get("output") or "").strip().splitlines()
        report["smoke"] = output[-1][:500] if output else ""
        report["passed"] = smoke.get("returncode") == 0 and '"status": "passed"' in (output[-1] if output else "")
    except Exception as error:  # noqa: BLE001 - reported, the check never trains
        report["exception"] = repr(error)[:2000]
        report["passed"] = False
    finally:
        environment.cleanup()
    report["seconds"] = round(time.time() - started, 1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import pandas as pd
    import yaml

    config = yaml.safe_load(args.harness.read_text())
    reports = []
    for row in pd.read_parquet(args.data).to_dict("records"):
        report = _check(json.loads(row["extra_info"]["instance_json"]), config)
        print(json.dumps(report), flush=True)
        reports.append(report)
    ok = all(report["passed"] for report in reports)
    if args.out:
        args.out.write_text(json.dumps({"passed": ok, "tasks": reports}, indent=2) + "\n")
    print(json.dumps({"passed": ok, "n": len(reports)}))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
