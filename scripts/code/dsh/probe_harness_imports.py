"""Instantiate both real harnesses without model requests or sandbox creation.

Catches lazy model/tool-registry imports that top-level package imports miss.
This CPU probe is not an executed-agent or training acceptance test.
"""

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace


def probe(source, configs):
    source = Path(source).resolve()
    sys.path[:0] = [str(source), str(source / "third_party/mimoagent-osr/src"), str(source / "third_party/uni_agent")]
    report = {
        "scope": "cpu_real_harness_constructor_no_model_request",
        "python": sys.executable,
        "source": str(source),
        "passed": False,
        "harnesses": [],
    }
    try:
        from mimoagent.agents.factory import get_agent_class

        from recipes.code.dsh_agent import DshSdkAgent
        from recipes.code.mimoagent_runner import _build_model, _load_config

        with tempfile.TemporaryDirectory(prefix="harness-import-probe-") as directory:
            env = SimpleNamespace(config=SimpleNamespace(cwd=directory))
            for kind, path in configs:
                path = Path(path)
                config = _load_config(path)
                agent_config = dict(config["agent"])
                actual_kind = agent_config.pop("type")
                if actual_kind != kind:
                    raise ValueError("unexpected harness type in probe config")
                agent_config.pop("msg_path", None)
                model = _build_model(config, "http://127.0.0.1:1/v1", agent_type=kind)
                agent_cls = DshSdkAgent if kind == "dsh-sdk" else get_agent_class(kind)
                agent = agent_cls(model, env, **agent_config)
                definitions = getattr(agent, "_tool_definitions", [])
                report["harnesses"].append(
                    {
                        "agent_type": kind,
                        "agent_class": f"{agent_cls.__module__}.{agent_cls.__name__}",
                        "model_class": f"{type(model).__module__}.{type(model).__name__}",
                        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "tool_names": [tool["function"]["name"] for tool in definitions],
                    }
                )
        report["tenacity_version"] = importlib.metadata.version("tenacity")
        report["passed"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["missing_module"] = getattr(exc, "name", None)
        report["frames"] = [
            {"file": f.filename, "line": f.lineno, "function": f.name} for f in traceback.extract_tb(exc.__traceback__)
        ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--mini-config", type=Path, required=True)
    parser.add_argument("--dsh-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = probe(args.source, [("mimocode-agent", args.mini_config), ("dsh-sdk", args.dsh_config)])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"passed": result["passed"], "harnesses": len(result["harnesses"]), "error_type": result.get("error_type")}
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
