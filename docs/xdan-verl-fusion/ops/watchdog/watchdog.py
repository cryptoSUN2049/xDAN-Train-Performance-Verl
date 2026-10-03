"""Training watchdog for the xDAN training lines, run on the operator Mac.

Every interval it SSH-polls each target pod, applies the alert rules and notifies through lark-cli (bot DM).
Every message starts with the target's title and goal and carries a progress bar.

- fusion-a: group A fusion RL (one long run). Health alerts, a summary every 5 steps, milestone events.
  --auto-stop (default off) stops it on a systemic failure: kills the trainer, stops Ray, terminates only
  this run's Modal app sandboxes.
- baseline: the baseline line's five-domain queue on its own pod. READ-ONLY: it never writes, stops or
  cleans anything there (that line self-heals with its own observe_active.py). Pushes each finished queue
  phase (with SFT/RL eval scores), queue death, long log silence, and a heartbeat every 6 hours.

Alerts fire once per condition and repeat at most hourly while it holds.
usage: python3 watchdog.py [--once] [--send] [--auto-stop] [--only fusion-a|baseline] [--run <fusion run dir>]
"""

# ruff: noqa: E501  (the remote probes are embedded Python sources sent over ssh)
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / "state.json"
LOG = HERE / "watchdog.log"
RECIPIENT_OPEN_ID = "ou_c09fa2f0ded5799efe08aaf4e750e113"
HOST = "root@157.157.221.177"
REPEAT_SECONDS = 3600

FUSION = {
    "name": "fusion-a",
    "title": "【A组·融合RL】group-a-r1",
    "port": 30293,
    "run": "/workspace/xdan-verl-fusion/runs/group-a-r1",
    "ckpt": "/workspace/xdan-verl-fusion/checkpoints/group-a-r1",
    "app": "xdan-fusion-group-a",
    "total_steps": 100,
    "goal": "TB2.1 或 Code holdout100（mean@4）至少一项比 SFT 高 ≥3pt，其余不退步；step 50 仍无增益则停",
    "milestones": {
        5: "首个 checkpoint（之后可断点续训；每 5 步存一次，保留最近 2 个）",
        25: "评测点 1：对比 SFT 基线（HF 权重已由 milestone_keeper 永久留存）",
        50: "决策点：对比 SFT，无增益则停",
        100: "第一轮完成 → 第二轮加入 batch1 的 1851 题",
    },
}
BASELINE = {
    "name": "baseline",
    "title": "【Baseline·五域复现】队列 6e",
    "port": 10924,
    "root": "/workspace/train-p0-dsh-integration",
    "summary": "runs/tier4h-20261002-summary.jsonl",
    "goal": "五域 4h 档：每域 SFT 评测 → 训练 N 步（真实更新 + 最终 checkpoint）→ RL 评测，"
    "给出 SFT/RL 配对差；缩比规模不要求涨分",
    "domains": ["music", "webdev", "general", "code", "cyber"],
    "tail_1h": [
        ("general", "train-1h-g3", "训练"),
        ("general", "resume-1h-g3", "恢复"),
        ("general", "rl-eval-1h", "RL评测"),
    ],
    "heartbeat_seconds": 6 * 3600,
    "stall_minutes": 45,
}

STALL_MINUTES = 100  # fusion: ~2x the measured step
STALL_IDLE_MINUTES = 30  # no trajectory finished for this long while stalled -> systemic
FAIL_WARN, FAIL_STOP = 0.05, 0.30
SANDBOX_WARN = 120
SUMMARY_EVERY = 5

FUSION_REMOTE = r"""
import json, os, re, subprocess, time, glob
P = json.loads(__PARAMS__)
R, CK = P["run"], P["ckpt"]
out = {"now": time.time()}
out["exit_code"] = open(R + "/exit-code.txt").read().strip() if os.path.exists(R + "/exit-code.txt") else None
steps = []
if os.path.exists(R + "/metrics.jsonl"):
    out["metrics_mtime"] = os.path.getmtime(R + "/metrics.jsonl")
    for line in open(R + "/metrics.jsonl"):
        m = json.loads(line); m = m.get("data", m)
        if "training/global_step" in m:
            steps.append({k: m.get(k) for k in ("training/global_step", "timing_s/step", "critic/rewards/mean",
                "actor/grad_norm", "harness/mimocode-agent/reward_mean", "harness/dsh-sdk/reward_mean",
                "dynsam/opensource-code/num_accepted/step", "dynsam/harbor/num_accepted/step")})
out["steps"] = steps
out["started"] = os.path.getmtime(R + "/started-utc.txt") if os.path.exists(R + "/started-utc.txt") else None
sessions = glob.glob(R + "/trajectories/step_*/session-*")
out["sessions_finished_recent"] = sum(1 for s in sessions if time.time() - os.path.getmtime(s) < 1800)
out["dsh_failures"] = len(glob.glob(R + "/trajectories/*/*/agent_msgs/dsh-failure-session.jsonl"))
# DSH turn-end reasons over the last 2 step dirs: "max-tokens" means a turn hit the per-turn output cap
recent_steps = sorted(glob.glob(R + "/trajectories/step_*"), key=lambda p: int(p.rsplit("_", 1)[1]))[-2:]
ends = {}
for d in recent_steps:
    for f in glob.glob(d + "/*/agent_msgs/dsh-session.jsonl"):
        with open(f, "rb") as fh:
            fh.seek(max(0, os.path.getsize(f) - 8192))
            tail = fh.read().decode("utf-8", "ignore")
        kinds = re.findall(r'"reason":\{"kind":"([a-z-]+)"\},"turn"', tail)  # turn/end events only
        if not kinds:
            continue
        kind = kinds[-1]
        if kind == "max-tokens":
            # DSH reports both the per-turn cap and a full 64K context as max-tokens; split them by usage
            usage = re.findall(r'"usage":\{"inputTokens":(\d+),"outputTokens":(\d+)', tail)
            kind = "turn-cap" if usage and int(usage[-1][1]) >= 30000 else "context-full"
        ends[kind] = ends.get(kind, 0) + 1
out["dsh_turn_end"] = ends
log = R + "/training.log"
text = open(log, errors="ignore").read()[-5_000_000:] if os.path.exists(log) else ""
out["ungradable"] = text.count("ungradable")
# one "generate_sequences summary" line per rollout batch; a failed session is dropped, a failed uid = whole group
summ = re.findall(r"num_success_sessions=(\d+) num_failed_sessions=(\d+).*?num_failed_uids=(\d+)", text)[-10:]
out["session_failures"] = [int(f) for _, f, _ in summ]
out["session_totals"] = [int(a) + int(f) for a, f, _ in summ]
out["failed_groups"] = sum(int(u) for _, _, u in summ)
out["fail_reasons"] = sorted(set(re.findall(r"failure_reasons=\['([A-Za-z]+Error)", text[-500_000:])))
out["ckpts"] = sorted(int(d.rsplit("_", 1)[1]) for d in os.listdir(CK) if d.startswith("global_step_")) if os.path.isdir(CK) else []
# fusion-eval writes runs/eval-a-<tag>-<harness>-<bench>/ (+ summary.json with strict = failed sessions count as 0);
# the TB2.1 baseline counts as done only when both harnesses finished. summary.json is written after the eval
# ends; metrics.jsonl is created empty at launch by verl FileLogger, so its existence proves nothing.
E = "/workspace/xdan-verl-fusion/runs"
out["sft_eval_done"] = all(os.path.exists(f"{E}/eval-a-sft-{h}-tb21/summary.json") for h in ("mimocode", "dsh"))
def strict(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if "strict" in k and isinstance(v, (int, float)):
                return v
            found = strict(v)
            if found is not None:
                return found
    return None
evals = {}
for d in sorted(glob.glob(E + "/eval-a-*")):
    name = os.path.basename(d)[len("eval-a-"):]
    if name.startswith("smoke") or not os.path.exists(d + "/summary.json"):
        continue
    try:
        evals[name] = strict(json.load(open(d + "/summary.json")))
    except Exception:
        evals[name] = None
out["evals"] = evals
origin = open("/workspace/xdan-verl-fusion/runs/dsh-gateway-runpod/public-origin.txt").read().strip()
probe = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-X", "POST",
    "-H", "Content-Type: application/json", "-d", "{}", "--max-time", "15",
    origin + "/sessions/probe/v1/chat/completions"], capture_output=True, text=True)
out["gateway_http"] = probe.stdout.strip()
# anchored: an unanchored pattern also matches the ssh command running this probe
out["trainer_alive"] = subprocess.run(["pgrep", "-f", "^python3 -m verl.trainer.main_ppo"], capture_output=True).returncode == 0
print(json.dumps(out))
"""

BASELINE_REMOTE = r"""
import json, os, re, subprocess, time, glob
P = json.loads(__PARAMS__)
R = P["root"]
out = {"now": time.time()}
def score(d):
    path = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(path):
        return None
    vals = []
    for line in open(path):
        m = json.loads(line); m = m.get("data", m)
        vals += [v for k, v in m.items() if re.match(r"val-core/.+/(reward|acc)/mean@\d+$", k) and isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None
recs = []
sp = os.path.join(R, P["summary"])
if os.path.exists(sp):
    for line in open(sp):
        if line.strip():
            r = json.loads(line)
            if "eval" in r.get("phase", ""):
                r["score"] = score(r["dir"])
            recs.append(r)
out["records"] = recs
plan = {}
for d in P["domains"]:
    try:
        plan[d] = json.load(open(f"{R}/data-tiers-20261002/{d}/4h/receipt.json"))["steps"]
    except Exception:
        plan[d] = None
out["steps_plan"] = plan
out["queue_alive"] = subprocess.run(["pgrep", "-f", "^bash gpu_queue"], capture_output=True).returncode == 0
qlogs = sorted(glob.glob(R + "/runs/gpu_queue*.log"), key=os.path.getmtime)
lines = open(qlogs[-1], errors="ignore").read().strip().splitlines() if qlogs else []
out["queue_tail"] = lines[-1][-200:] if lines else ""
cands = [os.path.dirname(p) for p in glob.glob(R + "/runs/*/started-utc.txt") + glob.glob(R + "/runs/*/*/started-utc.txt")]
cands = [d for d in cands if not os.path.exists(d + "/exit-code.txt") and time.time() - os.path.getmtime(d + "/started-utc.txt") < 86400]
act = None
if cands:
    a = max(cands, key=lambda d: os.path.getmtime(d + "/started-utc.txt"))
    lm = max((os.path.getmtime(a + f) for f in ("/training.log", "/driver.log") if os.path.exists(a + f)), default=None)
    step = 0
    if os.path.exists(a + "/metrics.jsonl"):
        for line in open(a + "/metrics.jsonl"):
            m = json.loads(line); m = m.get("data", m)
            if "training/global_step" in m:
                step = max(step, int(m["training/global_step"]))
    act = {"dir": a.replace(R + "/runs/", ""), "log_mtime": lm, "step": step, "started": os.path.getmtime(a + "/started-utc.txt")}
out["active"] = act
util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
out["gpu_util"] = [int(x) for x in util.stdout.split() if x.isdigit()]
print(json.dumps(out))
"""

ENV = r"""
P=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
source $P/activate.sh $P >/dev/null
source /workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/scripts/runtime.env
"""
LOOKUP = (
    'import modal; a = modal.App.lookup("__APP__", create_if_missing=False); '
    "boxes = modal.Sandbox.list(app_id=a.app_id)"
)
SANDBOXES = ENV + "timeout 90 python -c '" + LOOKUP + "; print(sum(1 for _ in boxes))'\n"
# Kill only the trainer; its bash -c wrapper then records exit-code.txt. Ray and sandboxes are this pod's own.
STOP = (
    'kill $(pgrep -f "^python3 -m verl.trainer.main_ppo")\n'
    + ENV
    + "ray stop --force >/dev/null 2>&1\n"
    + "timeout 300 python -c '"
    + LOOKUP
    + "; [s.terminate() for s in boxes]'\necho stopped\n"
)


def log(line: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with LOG.open("a") as stream:
        stream.write(f"{stamp} {line}\n")


def remote(port: int, script: str, *, python: bool) -> str:
    ssh = ["ssh", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", "-p", str(port)]
    ssh += ["-i", str(Path.home() / ".ssh/id_ed25519"), HOST]
    result = subprocess.run(
        ssh + (["python3", "-"] if python else ["bash", "-s"]),
        input=script,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ssh :{port} rc={result.returncode}: {result.stderr.strip()[-300:]}")
    return result.stdout.strip().splitlines()[-1]


def with_params(script: str, params: dict) -> str:
    return script.replace("__PARAMS__", repr(json.dumps(params)))


def bar(done: float, total: float, width: int = 20) -> str:
    frac = 0 if total <= 0 else max(0.0, min(1.0, done / total))
    filled = round(frac * width)
    return f"`{'▓' * filled}{'░' * (width - filled)}` {done:g}/{total:g}（{frac:.0%}）"


def notify(target: dict, body: str, send: bool, *, icon: str = "") -> None:
    markdown = f"## {icon}{target['title']}\n**目标**：{target['goal']}\n\n{body}"
    log(f"NOTIFY [{target['name']}] " + body.replace("\n", " | ")[:600])
    if not send:
        return
    result = subprocess.run(
        ["lark-cli", "im", "+messages-send", "--as", "bot", "--user-id", RECIPIENT_OPEN_ID, "--markdown", markdown],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1", "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1"},
    )
    if result.returncode != 0:
        log(f"NOTIFY FAILED rc={result.returncode}: {result.stderr.strip()[-400:]}")


def fire(target: dict, state: dict, alerts: list[tuple[str, str]], send: bool, footer: str) -> None:
    """Send each alert once, repeating at most hourly while it holds; forget alerts that cleared."""
    now, fired = time.time(), state.setdefault("fired", {})
    for key in list(fired):
        if key not in {k for k, _ in alerts} and key != "ssh":
            fired.pop(key)
    for key, text in alerts:
        if now - fired.get(key, 0) > REPEAT_SECONDS:
            notify(target, f"**报警**：{text}\n\n{footer}", send, icon="⚠️ ")
            fired[key] = now


# ---------------------------------------------------------------- fusion-a


def fusion_alerts(s: dict, state: dict) -> tuple[list[tuple[str, str, bool]], float]:
    """Return (alerts as (key, text, systemic)), minutes since the last step."""
    alerts = []
    idle_min = (s["now"] - (s.get("metrics_mtime") or s.get("started") or s["now"])) / 60
    if s.get("exit_code") is not None:
        alerts.append(("exited", f"训练进程已退出，exit-code={s['exit_code']}（完成或崩溃，请检查）", False))
    elif not s.get("trainer_alive"):
        alerts.append(("dead", "训练进程不在运行，但没有 exit-code（可能被杀或 pod 重启）", False))
    if s.get("exit_code") is None and idle_min > STALL_MINUTES:
        systemic = s.get("sessions_finished_recent", 0) == 0 and idle_min > STALL_MINUTES + STALL_IDLE_MINUTES
        alerts.append(
            (
                "stall",
                f"已 {idle_min:.0f} 分钟没有新的训练步（阈值 {STALL_MINUTES}）；"
                f"近 30 分钟完成轨迹 {s.get('sessions_finished_recent', 0)}",
                systemic,
            )
        )
    failures, totals = s.get("session_failures") or [], s.get("session_totals") or []
    if sum(totals) >= 32:
        ratio = sum(failures) / sum(totals)
        detail = f"（失败 {sum(failures)}/{sum(totals)} 条，整组失败 {s.get('failed_groups')}，"
        detail += f"类型 {s.get('fail_reasons')}）"
        if ratio > FAIL_STOP:
            alerts.append(("fail_stop", f"最近 rollout 失败率 {ratio:.0%} > {FAIL_STOP:.0%}{detail}", True))
        elif ratio > FAIL_WARN:
            alerts.append(("fail_warn", f"最近 rollout 失败率 {ratio:.0%} > {FAIL_WARN:.0%}{detail}", False))
    new_dsh = s.get("dsh_failures", 0) - state.get("dsh_failures_seen", 0)
    if new_dsh >= 3:
        alerts.append(("dsh", f"新增 DSH 失败 {new_dsh} 条（累计 {s.get('dsh_failures')}）", False))
    ends = s.get("dsh_turn_end") or {}
    if sum(ends.values()) >= 16 and ends.get("turn-cap", 0) / sum(ends.values()) > 0.10:
        alerts.append(("dsh_cap", f"DSH 会话撞单轮输出上限（32768）比例偏高：{ends}（应接近 0）", False))
    if s.get("gateway_http") != "401":
        alerts.append(("gateway", f"DSH 网关探测异常：HTTP {s.get('gateway_http')}（预期 401）", False))
    if (s.get("sandboxes") or 0) > SANDBOX_WARN:
        alerts.append(("sandbox", f"Modal 沙箱 {s['sandboxes']} 个（>{SANDBOX_WARN}，可能泄漏）", False))
    return alerts, idle_min


def evals_text(evals: dict | None) -> str:
    if not evals:
        return "无"
    return "，".join(f"{k} {'-' if v is None else format(v, '.3f')}" for k, v in evals.items())


def fusion_progress(s: dict, done: int) -> str:
    total, milestones = FUSION["total_steps"], FUSION["milestones"]
    upcoming = [m for m in sorted(milestones) if m > done]
    nxt = f"step {upcoming[0]}：{milestones[upcoming[0]]}" if upcoming else "全部节点已过"
    return (
        f"**进度** {bar(done, total)}\n"
        f"- 下一节点：{nxt}\n"
        f"- SFT 基线（TB2.1，两个 harness）：{'✅ 已完成' if s.get('sft_eval_done') else '⏳ 未完成（step 50 判定需要它）'}\n"
        f"- 已完成评测（strict）：{evals_text(s.get('evals'))}\n"
        f"- 已保存 checkpoint：{s.get('ckpts') or '无'}"
    )


def fusion_summary(s: dict) -> str:
    steps = s.get("steps") or []
    recent = steps[-SUMMARY_EVERY:]
    last = steps[-1]["training/global_step"]

    def avg(key):
        values = [x.get(key) for x in recent if x.get(key) is not None]
        return sum(values) / len(values) if values else None

    def fmt(value, spec=".3f"):
        return "-" if value is None else format(value, spec)

    code = sum(x.get("dynsam/opensource-code/num_accepted/step") or 0 for x in recent)
    harbor = sum(x.get("dynsam/harbor/num_accepted/step") or 0 for x in recent)
    remaining = (FUSION["total_steps"] - last) * (avg("timing_s/step") or 0) / 3600
    return (
        f"{fusion_progress(s, last)}\n\n"
        f"**近 {len(recent)} 步**\n"
        f"- 步时 {fmt((avg('timing_s/step') or 0) / 60, '.1f')} 分钟，预计剩余 {remaining:.0f} 小时\n"
        f"- reward {fmt(avg('critic/rewards/mean'))}，grad {fmt(avg('actor/grad_norm'))}\n"
        f"- harness：mimocode {fmt(avg('harness/mimocode-agent/reward_mean'))} / "
        f"DSH {fmt(avg('harness/dsh-sdk/reward_mean'))}\n"
        f"- 被接收的组：Code {code} / Harbor {harbor}\n"
        f"- DSH 结束原因（近 2 步）：{s.get('dsh_turn_end') or '-'}\n"
        f"- DSH 失败累计 {s.get('dsh_failures')}，ungradable {s.get('ungradable')}，沙箱 {s.get('sandboxes')}"
    )


def tick_fusion(state: dict, run: str, send: bool, auto_stop: bool) -> str:
    target = FUSION
    params = {"run": run, "ckpt": target["ckpt"]}
    snapshot = json.loads(remote(target["port"], with_params(FUSION_REMOTE, params), python=True))
    try:
        snapshot["sandboxes"] = int(remote(target["port"], SANDBOXES.replace("__APP__", target["app"]), python=False))
    except Exception as error:  # noqa: BLE001
        snapshot["sandboxes"] = None
        log(f"WARN sandbox count failed: {error}")
    alerts, idle_min = fusion_alerts(snapshot, state)
    done = (snapshot.get("steps") or [{}])[-1].get("training/global_step") or 0
    footer = f"**进度** {bar(done, target['total_steps'])}，距上一步 {idle_min:.0f} 分钟"
    fire(target, state, [(k, t) for k, t, _ in alerts], send, footer)
    systemic = [a for a in alerts if a[2]]
    if systemic and auto_stop and not state.get("stopped"):
        result = remote(target["port"], STOP.replace("__APP__", target["app"]), python=False)
        state["stopped"] = time.time()
        reason = "；".join(a[1] for a in systemic)
        notify(
            target,
            f"**已自动停止**\n\n原因：{reason}\n\n结果：{result}。已回收本 run 的 Modal 沙箱，修复后从最近 checkpoint 恢复。",
            send,
            icon="🛑 ",
        )
    for m in sorted(target["milestones"]):
        if state.get("milestone_seen", 0) < m <= done:
            notify(
                target, f"**到达节点 step {m}**：{target['milestones'][m]}\n\n{fusion_progress(snapshot, done)}", send
            )
            state["milestone_seen"] = m
    if done and done % SUMMARY_EVERY == 0 and state.get("summary_step") != done:
        notify(target, fusion_summary(snapshot), send)
        state["summary_step"] = done
    state["dsh_failures_seen"] = snapshot.get("dsh_failures", 0)
    fails = f"{sum(snapshot.get('session_failures') or [])}/{sum(snapshot.get('session_totals') or [])}"
    return (
        f"steps={done} idle={idle_min:.0f}m sandboxes={snapshot.get('sandboxes')} dsh_fail={snapshot.get('dsh_failures')} "
        f"sess_fail={fails} gateway={snapshot.get('gateway_http')} alerts={[a[0] for a in alerts]}"
    )


# ---------------------------------------------------------------- baseline (read-only)

PHASES_4H = [("sft-eval", "SFT评测"), ("train", "训练"), ("rl-eval", "RL评测")]


def node_key(record: dict) -> tuple[str, str]:
    phase = record["phase"]
    return record["domain"], "train" if phase == "grader_up" else phase


def baseline_progress(s: dict) -> str:
    target, records = BASELINE, s.get("records") or []
    done = {node_key(r): r for r in records}
    plan = s.get("steps_plan") or {}

    def mark(rec, label, extra=""):
        if rec is None:
            return f"⏳{label}"
        return f"{'✅' if str(rec.get('rc')) == '0' else '❌'}{label}{extra}"

    def scored(rec):
        return f" {rec['score']:.3f}" if rec and rec.get("score") is not None else ""

    total = len(target["tail_1h"]) + len(target["domains"]) * len(PHASES_4H)
    finished = sum(1 for d, p, _ in target["tail_1h"] if (d, p) in done)
    tail = " · ".join(mark(done.get((d, p)), label) for d, p, label in target["tail_1h"])
    lines = [f"- 1h 收尾（General g3）：{tail}"]
    for d in target["domains"]:
        sft, train, rl = (done.get((d, p)) for p, _ in PHASES_4H)
        finished += sum(1 for r in (sft, train, rl) if r)
        delta = ""
        if sft and rl and sft.get("score") is not None and rl.get("score") is not None:
            delta = f"（RL−SFT {rl['score'] - sft['score']:+.3f}）"
        n = plan.get(d)
        parts = [mark(sft, "SFT", scored(sft)), mark(train, f"训练{n or '?'}步"), mark(rl, "RL", scored(rl) + delta)]
        lines.append(f"- {d.capitalize()}：" + " · ".join(parts))
    act = s.get("active")
    if act:
        current = act["dir"]
        if act.get("step"):
            current += f"（step {act['step']}）"
        mins = (s["now"] - act["started"]) / 60
        lines.append(f"- 当前：{current}，已运行 {mins:.0f} 分钟，GPU 利用率 {s.get('gpu_util')}")
    else:
        lines.append(f"- 当前：无运行中的任务；队列尾行：{s.get('queue_tail')}")
    return f"**进度** {bar(finished, total)}\n" + "\n".join(lines)


def tick_baseline(state: dict, send: bool) -> str:
    target = BASELINE
    params = {k: target[k] for k in ("root", "summary", "domains")}
    s = json.loads(remote(target["port"], with_params(BASELINE_REMOTE, params), python=True))
    records = s.get("records") or []
    seen = state.get("records_seen")
    if seen is None:  # first run: announce the current state once instead of replaying history
        notify(target, "看门狗已接入（只读，不会停止或清理任何东西）。\n\n" + baseline_progress(s), send)
        state["heartbeat"] = time.time()
        seen = len(records)
    for record in records[seen:]:
        ok = str(record.get("rc")) == "0"
        score = f"，得分 {record['score']:.3f}" if record.get("score") is not None else ""
        status = "完成" if ok else f"失败 rc={record.get('rc')}"
        head = f"**{record['domain']} · {record['phase']}** {status}{score}"
        notify(target, f"{head}\n\n{baseline_progress(s)}", send, icon="" if ok else "⚠️ ")
    state["records_seen"] = len(records)
    alerts = []
    finished = bool(re.search(r"queue\w* done$", s.get("queue_tail") or ""))  # "queue6 done", not "... fresh done rc=0"
    if not s.get("queue_alive"):
        if finished and not state.get("finished_sent"):
            notify(target, "**队列全部结束**\n\n" + baseline_progress(s), send, icon="🏁 ")
            state["finished_sent"] = True
        elif not finished:
            alerts.append(("queue_dead", f"队列进程不在了，但队列没有结束。尾行：{s.get('queue_tail')}"))
    act = s.get("active")
    if s.get("queue_alive") and act and act.get("log_mtime"):
        idle = (s["now"] - act["log_mtime"]) / 60
        if idle > target["stall_minutes"]:
            alerts.append(("stall", f"{act['dir']} 已 {idle:.0f} 分钟没有写日志（阈值 {target['stall_minutes']}）"))
    fire(target, state, alerts, send, baseline_progress(s))
    if time.time() - state.get("heartbeat", 0) > target["heartbeat_seconds"]:
        notify(target, "**定时进度**\n\n" + baseline_progress(s), send)
        state["heartbeat"] = time.time()
    return f"records={len(records)} queue_alive={s.get('queue_alive')} active={(act or {}).get('dir')} alerts={alerts}"


# ---------------------------------------------------------------- loop


def tick(only: str | None, run: str, send: bool, auto_stop: bool) -> None:
    states = json.loads(STATE.read_text()) if STATE.exists() else {}
    if "fired" in states:  # single-target state from the first version
        states = {"fusion-a": states}
    jobs = {
        "fusion-a": lambda st: tick_fusion(st, run, send, auto_stop),
        "baseline": lambda st: tick_baseline(st, send),
    }
    targets = {"fusion-a": FUSION, "baseline": BASELINE}
    for name, job in jobs.items():
        if only and name != only:
            continue
        state = states.setdefault(name, {})
        try:
            log(f"OK [{name}] {job(state)}")
        except Exception as error:  # noqa: BLE001
            log(f"ERROR [{name}] {error}")
            fired = state.setdefault("fired", {})
            if time.time() - fired.get("ssh", 0) > REPEAT_SECONDS:
                notify(targets[name], f"**看门狗无法读取状态**：{str(error)[:300]}", send, icon="⚠️ ")
                fired["ssh"] = time.time()
        else:
            state.get("fired", {}).pop("ssh", None)
        STATE.write_text(json.dumps(states))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--send", action="store_true", help="send Feishu messages (default: log only)")
    parser.add_argument("--auto-stop", action="store_true", help="fusion-a only")
    parser.add_argument("--only", choices=["fusion-a", "baseline"])
    parser.add_argument(
        "--run", default=FUSION["run"], help="fusion run dir (metrics.jsonl, training.log, exit-code.txt)"
    )
    parser.add_argument("--interval", type=int, default=600)
    args = parser.parse_args()
    while True:
        tick(args.only, args.run, args.send, args.auto_stop)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
