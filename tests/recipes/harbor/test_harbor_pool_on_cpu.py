# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""CPU tests for full-pool task selection and the nop/oracle audit rule."""

from __future__ import annotations

from scripts.harbor.oracle_check import audit_passed
from scripts.harbor.select_tasks import select


def _row(source, task, *, split="train", status="derived", repo="r"):
    return {"source": source, "task": task, "split": split, "status": status, "category": repo, "task_dir": task}


def test_select_respects_split_status_exclusions_and_quotas():
    rows = [_row("tl", f"t{i}") for i in range(10)]
    rows += [_row("tl", "val", split="validation"), _row("tl", "bad", status="needs_manual_review")]
    chosen = select(rows, {"tl": 5}, excluded={("tl", "t0"), ("tl", "t1")}, max_per_repo=30)
    names = {r["task"] for r in chosen}
    assert len(chosen) == 5
    assert not names & {"t0", "t1", "val", "bad"}


def test_select_is_deterministic_and_caps_swe_repositories():
    rows = [_row("swe-x", f"a{i}", repo="big") for i in range(10)] + [
        _row("swe-x", f"b{i}", repo="small") for i in range(3)
    ]
    first = select(rows, {"swe-x": 8}, excluded=set(), max_per_repo=4)
    again = select(list(reversed(rows)), {"swe-x": 8}, excluded=set(), max_per_repo=4)
    assert [r["task"] for r in first] == [r["task"] for r in again]
    assert sum(r["category"] == "big" for r in first) == 4
    assert len(first) == 7  # 4 from the capped repository + all 3 others


def test_audit_requires_nop_zero_and_oracle_one():
    good = {"untouched": {"reward": 0.0, "error_category": None}, "solved": {"reward": 1.0, "error_category": None}}
    assert audit_passed(good)
    assert not audit_passed({**good, "untouched": {"reward": 1.0, "error_category": None}})  # trivially solved
    assert not audit_passed({**good, "solved": {"reward": 0.0, "error_category": None}})  # oracle fails
    assert not audit_passed({**good, "untouched": {"reward": 0.0, "error_category": "harbor_reward_missing"}})
    assert not audit_passed({**good, "exception": "boom"})
