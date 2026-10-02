"""Compose each recipe config and instantiate every node that declares ``_target_``.

``main_ppo --cfg job`` only proves Hydra composition; workers additionally build the config
dataclasses, which is where removed fields (e.g. ``grad_offload``) fail. Run from a repo root:

    python docs/xdan-verl-fusion/ops/dataclass_check.py

Compare the failure set against pristine mimo-oss: only failures new on the fusion branch matter
(the remaining ones come from local environment gaps such as missing model paths).
"""

import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from verl.utils.config import omega_conf_to_dataclass

RECIPES = [
    ("recipes/code/config", "train"),
    ("recipes/design/config", "music"),
    ("recipes/design/config", "webdev"),
    ("recipes/general/config", "general"),
    ("recipes/arvo/config", "arvo"),
]


def target_paths(node, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, DictConfig):
        if "_target_" in node:
            found.append(path)
        for key in node.keys():
            try:
                child = node._get_node(key)
            except Exception:  # noqa: BLE001 - unresolvable interpolations are skipped
                continue
            found.extend(target_paths(child, f"{path}.{key}" if path else str(key)))
    return found


def root_cause(error: BaseException) -> BaseException:
    while error.__cause__ is not None:
        error = error.__cause__
    return error


def main() -> int:
    bad = 0
    for config_dir, name in RECIPES:
        with initialize_config_dir(config_dir=os.path.join(os.getcwd(), config_dir), version_base=None):
            cfg = compose(config_name=name)
        targets = target_paths(cfg)
        outermost = [p for p in targets if not any(p != q and p.startswith(q + ".") for q in targets)]
        failures = []
        for path in outermost:
            try:
                omega_conf_to_dataclass(OmegaConf.select(cfg, path))
            except Exception as error:  # noqa: BLE001 - reported per node
                cause = root_cause(error)
                failures.append((path, f"{type(cause).__name__}: {str(cause).splitlines()[0][:200]}"))
        print(f"{name}: {len(targets)} targets, {len(failures)} failures")
        for path, message in failures:
            print(f"   FAIL {path}: {message}")
        bad += len(failures)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
