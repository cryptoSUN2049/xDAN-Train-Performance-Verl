# Copyright 2026 xDAN contributors
# Licensed under the Apache License, Version 2.0.
"""Relabel Harbor training tasks into the 16-category TB2.1 taxonomy with an LLM.

Each task directory holds ``task.toml`` (native ``metadata.category`` / ``difficulty`` / ``tags``) and
``instruction.md``. The LLM sees the category names with one-line definitions written for this script
(no TB2.1 task text, TB2.1 is an evaluation set), the native metadata and the (truncated) instruction,
and returns ``{category, secondary_category, confidence, rationale, difficulty_hint}``.

Results are appended to a JSONL cache keyed by ``(instance_id, prompt_version)`` so reruns resume.
Endpoint: an OpenAI-compatible proxy from ``LITELLM_BASE_URL`` / ``LITELLM_API_KEY`` (never logged).

usage: relabel_tb21_taxonomy.py --tasks-dir batch1/tasks --ids audit-summary.json --cache labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tomllib

PROMPT_VERSION = "v2"

# One-line definitions written for this script; they describe the skill being exercised, not any eval task.
CATEGORIES: dict[str, str] = {
    "software-engineering": "Build, extend or refactor a program, library, service or CLI to a specification "
    "(new features, APIs, integrations, build systems); the main work is writing new code.",
    "system-administration": "Install, configure, operate or troubleshoot an OS environment: services, packages, "
    "users/permissions, cron, networking, web servers, containers, shell environment.",
    "scientific-computing": "Numerical or domain-science computation: simulations, numerical methods, physics/"
    "chemistry/biology/astronomy models, scientific file formats and libraries.",
    "security": "Vulnerabilities, exploitation or hardening, cryptography, authentication/secrets handling, "
    "sanitizing input, forensics, password or hash recovery.",
    "data-science": "Analyse a dataset to answer questions: exploratory analysis, statistics, aggregation, "
    "plotting or reporting insights (pandas/numpy-style analytics).",
    "debugging": "Diagnose and fix existing code or an environment that is broken (crash, wrong output, failing "
    "test, regression); the main work is locating the root cause of a reported defect.",
    "file-operations": "Manipulate files and directories: find, rename, move, archive/extract, convert encodings "
    "or formats, recover or diff files, bulk filesystem edits.",
    "model-training": "Train or fine-tune a machine-learning model end to end: data loading, training loop, "
    "hyperparameters, checkpoints, reaching a target metric.",
    "mathematics": "Problems whose core is mathematics: proofs, symbolic or exact computation, number theory, "
    "combinatorics, geometry, probability puzzles.",
    "data-processing": "ETL on structured or semi-structured data: parse, clean, validate, transform, merge or "
    "convert CSV/JSON/XML/log records into a required output.",
    "machine-learning": "Use ML models or libraries without a full training run: inference, model loading or "
    "conversion, embeddings, evaluation metrics, feature engineering, small sklearn fits.",
    "games": "Implement, simulate, play or solve a game or game-like puzzle (board games, chess, puzzles, "
    "game engines, game-playing agents).",
    "personal-assistant": "Everyday user-productivity chores: scheduling, calendars, email, contacts, personal "
    "documents, organising notes or records on someone's behalf.",
    "optimization": "Make something faster or cheaper (profiling, algorithmic speed-up, memory reduction) or "
    "solve a formal optimisation problem (LP/IP, scheduling, constraint satisfaction).",
    "data-querying": "Retrieve answers from a data store by writing queries: SQL, NoSQL, GraphQL, SPARQL, "
    "search indexes; schema-aware querying is the core skill.",
    "video-processing": "Process audio, video or image media: decode, transcode, extract frames or metadata, "
    "ffmpeg pipelines, computer vision on media files.",
}
DIFFICULTIES = ("easy", "medium", "hard")

SYSTEM_PROMPT = (
    "You classify terminal/coding agent tasks into a fixed taxonomy. Pick the category whose definition "
    "best matches the DOMINANT skill the agent must exercise to pass the task's tests.\n\nCategories:\n"
    + "\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items())
    + "\n\nRules:\n"
    "- The programming language alone never decides the category.\n"
    "- Prefer the most specific domain category when the domain is central (e.g. writing SQL -> data-querying, "
    "cryptography -> security, ffmpeg -> video-processing, configuring nginx -> system-administration).\n"
    "- Resolving a GitHub issue in a repository is software-engineering when it asks for new behaviour or a "
    "feature/enhancement, and debugging when it reports a defect (crash, wrong result, regression) to fix.\n"
    "- secondary_category is another category that also clearly applies, else null.\n"
    "- difficulty_hint is your own estimate for a strong expert agent: easy = a few commands or one small "
    "function; medium = several steps or a moderate code change; hard = multi-component, deep domain knowledge "
    "or long debugging. Ignore any difficulty given in the metadata.\n"
    'Answer with ONLY a JSON object: {"category": str, "secondary_category": str|null, "confidence": number '
    '0-1, "rationale": str (<= 20 words), "difficulty_hint": "easy"|"medium"|"hard"}'
)


def load_task(task_dir: Path, max_chars: int) -> dict:
    meta = tomllib.loads((task_dir / "task.toml").read_text()).get("metadata", {})
    instruction = (task_dir / "instruction.md").read_text(errors="replace")
    if len(instruction) > max_chars:
        instruction = instruction[:max_chars] + "\n...[truncated]"
    return {
        "instance_id": task_dir.name,
        "family": task_dir.name.split("__", 1)[0],
        "native_category": meta.get("category"),
        "native_difficulty": meta.get("difficulty"),
        "tags": list(meta.get("tags", []) or []),
        "instruction": instruction,
    }


def user_prompt(task: dict) -> str:
    return (
        f"Native category: {task['native_category']}\nNative tags: {', '.join(task['tags']) or '-'}\n\n"
        f"Task instruction:\n<<<\n{task['instruction']}\n>>>"
    )


def parse_label(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON object in response")
    obj = json.loads(match.group(0))
    category = str(obj.get("category", "")).strip().lower()
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    secondary = obj.get("secondary_category")
    secondary = str(secondary).strip().lower() if secondary else None
    if secondary not in CATEGORIES or secondary == category:
        secondary = None
    hint = str(obj.get("difficulty_hint", "")).strip().lower()
    if hint not in DIFFICULTIES:
        raise ValueError(f"bad difficulty_hint {hint!r}")
    confidence = min(1.0, max(0.0, float(obj.get("confidence", 0.0))))
    rationale = " ".join(str(obj.get("rationale", "")).split()[:20])
    return {
        "category": category,
        "secondary_category": secondary,
        "confidence": confidence,
        "rationale": rationale,
        "difficulty_hint": hint,
    }


def chat(base_url: str, api_key: str, model: str, messages: list[dict], timeout: float) -> dict:
    body = json.dumps({"model": model, "messages": messages, "temperature": 0, "max_tokens": 2048}).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def classify(task: dict, args: argparse.Namespace, base_url: str, api_key: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(task)}]
    last_error = ""
    for attempt in range(args.retries):
        model = args.model if attempt < args.retries - 2 else args.fallback_model
        try:
            response = chat(base_url, api_key, model, messages, args.timeout)
            content = response["choices"][0]["message"].get("content") or ""
            label = parse_label(content)
            usage = response.get("usage") or {}
            return {
                "instance_id": task["instance_id"],
                "family": task["family"],
                "native_category": task["native_category"],
                "native_difficulty": task["native_difficulty"],
                "tags": task["tags"],
                **label,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "attempts": attempt + 1,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            time.sleep(min(30.0, 2.0**attempt) + random.random())
    raise RuntimeError(f"{task['instance_id']}: gave up after {args.retries} attempts ({last_error})")


def load_cache(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("prompt_version") == PROMPT_VERSION:
                    done[row["instance_id"]] = row
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--ids", type=Path, help="audit-summary.json (passed_ids) or a newline id list")
    parser.add_argument("--cache", type=Path, required=True, help="JSONL cache / output")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--fallback-model", default="deepseek-v4-pro")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=0, help="only label the first N pending ids (smoke)")
    args = parser.parse_args()
    if args.concurrency > 8:
        parser.error("concurrency must be <= 8")

    base_url, api_key = os.environ.get("LITELLM_BASE_URL"), os.environ.get("LITELLM_API_KEY")
    if not base_url or not api_key:
        raise SystemExit("LITELLM_BASE_URL / LITELLM_API_KEY not set (source ~/.zshenv)")

    if args.ids and args.ids.suffix == ".json":
        ids = json.loads(args.ids.read_text())["passed_ids"]
    elif args.ids:
        ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    else:
        ids = sorted(p.name for p in args.tasks_dir.iterdir() if p.is_dir())
    done = load_cache(args.cache)
    pending = [i for i in ids if i not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"ids={len(ids)} cached={len(ids) - len(set(ids) - set(done))} pending={len(pending)}", file=sys.stderr)

    lock, failures = threading.Lock(), []
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("a") as sink, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(classify, load_task(args.tasks_dir / i, args.max_chars), args, base_url, api_key): i
            for i in pending
        }
        for n, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
            except RuntimeError as exc:
                failures.append(str(exc))
                continue
            with lock:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
            if n % 50 == 0:
                print(f"{n}/{len(pending)} done, failures={len(failures)}", file=sys.stderr)
    if failures:
        print("\n".join(failures[:20]), file=sys.stderr)
        raise SystemExit(f"{len(failures)} tasks failed; rerun to resume")
    print(f"labelled {len(pending)} tasks -> {args.cache}", file=sys.stderr)


if __name__ == "__main__":
    main()
