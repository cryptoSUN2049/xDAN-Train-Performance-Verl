"""Training watchdog for the fusion pod, run on the operator Mac.

Every interval: SSH to the GPU pod, collect a status snapshot of the formal run, apply the alert rules,
and notify through lark-cli (bot DM). Alerts fire once per condition and repeat at most hourly while it
holds; a summary is sent every N completed steps. AUTO_STOP (default off) stops the run on a systemic
failure: kills the trainer, stops Ray and terminates only this run's Modal sandboxes (own app name).

usage: python3 watchdog.py [--once] [--send] [--auto-stop]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

POD = [
    "ssh",
    "-o",
    "ConnectTimeout=20",
    "-o",
    "ServerAliveInterval=30",
    "-p",
    "30293",
    "-i",
    str(Path.home() / ".ssh/id_ed25519"),
    "root@157.157.221.177",
]
RUN = "/workspace/xdan-verl-fusion/runs/group-a-r1"
MODAL_APP = "xdan-fusion-group-a"
RECIPIENT_OPEN_ID = "ou_c09fa2f0ded5799efe08aaf4e750e113"
STATE = Path(__file__).with_name("state.json")
LOG = Path(__file__).with_name("watchdog.log")

STALL_MINUTES = 100  # ~2x the measured 50-minute step
STALL_IDLE_MINUTES = 30  # no trajectory finished for this long while stalled -> systemic
FAIL_WARN, FAIL_STOP = 0.05, 0.30
SANDBOX_WARN = 120
SUMMARY_EVERY = 5
REPEAT_SECONDS = 3600

REMOTE = r"""
import json, os, re, subprocess, time, glob
R = "%(run)s"
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
                "dynsam/opensource-code/num_accepted/step", "dynsam/harbor/num_accepted/step",
                "training/filter_groups/evicted_samples", "training/rollout_failure/evicted_samples",
                "response_length/mean")})
out["steps"] = steps
out["started"] = os.path.getmtime(R + "/started-utc.txt") if os.path.exists(R + "/started-utc.txt") else None
sessions = glob.glob(R + "/trajectories/step_*/session-*")
out["sessions"] = len(sessions)
out["sessions_finished_recent"] = sum(1 for s in sessions if time.time() - os.path.getmtime(s) < 1800)
out["dsh_failures"] = len(glob.glob(R + "/trajectories/*/*/agent_msgs/dsh-failure-session.jsonl"))
log = R + "/training.log"
text = open(log, errors="ignore").read()[-5_000_000:] if os.path.exists(log) else ""
out["ungradable"] = text.count("ungradable")
# one "generate_sequences summary" line per rollout batch; a failed session is dropped, a failed uid = whole group
summ = re.findall(r"num_success_sessions=(\d+) num_failed_sessions=(\d+).*?num_failed_uids=(\d+)", text)[-10:]
out["session_failures"] = [int(f) for _, f, _ in summ]
out["session_totals"] = [int(a) + int(f) for a, f, _ in summ]
out["failed_groups"] = sum(int(u) for _, _, u in summ)
out["fail_reasons"] = sorted(set(re.findall(r"failure_reasons=\['([A-Za-z]+Error)", text[-500_000:])))
origin = open("/workspace/xdan-verl-fusion/runs/dsh-gateway-runpod/public-origin.txt").read().strip()
probe = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%%{http_code}", "-X", "POST",
    "-H", "Content-Type: application/json", "-d", "{}", "--max-time", "15",
    origin + "/sessions/probe/v1/chat/completions"], capture_output=True, text=True)
out["gateway_http"] = probe.stdout.strip()
# anchored: an unanchored pattern also matches the ssh command running this probe
alive = subprocess.run(["pgrep", "-f", "^python3 -m verl.trainer.main_ppo"], capture_output=True)
out["trainer_alive"] = alive.returncode == 0
print(json.dumps(out))
"""

ENV = r"""
P=/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654
source $P/activate.sh $P >/dev/null
source /workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/scripts/runtime.env
"""
LOOKUP = (
    'import modal; a = modal.App.lookup("%(app)s", create_if_missing=False); '
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


def remote(script: str, *, python: bool) -> str:
    command = POD + (["python3", "-"] if python else ["bash", "-s"])
    result = subprocess.run(command, input=script, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ssh rc={result.returncode}: {result.stderr.strip()[-300:]}")
    return result.stdout.strip().splitlines()[-1]


def notify(markdown: str, send: bool) -> None:
    log("NOTIFY " + markdown.replace("\n", " | "))
    if not send:
        return
    result = subprocess.run(
        ["lark-cli", "im", "+messages-send", "--as", "bot", "--user-id", RECIPIENT_OPEN_ID, "--markdown", markdown],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **__import__("os").environ,
            "LARKSUITE_CLI_NO_UPDATE_NOTIFIER": "1",
            "LARKSUITE_CLI_NO_SKILLS_NOTIFIER": "1",
        },
    )
    if result.returncode != 0:
        log(f"NOTIFY FAILED rc={result.returncode}: {result.stderr.strip()[-400:]}")


def evaluate(s: dict, state: dict) -> tuple[list[tuple[str, str, bool]], dict]:
    """Return (alerts as (key, text, systemic)), facts for the summary."""
    alerts = []
    now = s["now"]
    last_progress = s.get("metrics_mtime") or s.get("started") or now
    idle_min = (now - last_progress) / 60
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
    if s.get("gateway_http") not in ("401",):
        alerts.append(("gateway", f"DSH 网关探测异常：HTTP {s.get('gateway_http')}（预期 401）", False))
    if (s.get("sandboxes") or 0) > SANDBOX_WARN:
        alerts.append(("sandbox", f"Modal 沙箱 {s['sandboxes']} 个（>{SANDBOX_WARN}，可能泄漏）", False))
    return alerts, {"idle_min": idle_min}


def summary(s: dict) -> str:
    steps = s.get("steps") or []
    last = steps[-1]
    recent = steps[-SUMMARY_EVERY:]

    def avg(key):
        values = [x.get(key) for x in recent if x.get(key) is not None]
        return sum(values) / len(values) if values else None

    def fmt(value, spec=".3f"):
        return "-" if value is None else format(value, spec)

    code = sum(x.get("dynsam/opensource-code/num_accepted/step") or 0 for x in recent)
    harbor = sum(x.get("dynsam/harbor/num_accepted/step") or 0 for x in recent)
    remaining = (100 - last["training/global_step"]) * (avg("timing_s/step") or 0) / 3600
    return (
        f"## group-a-r1 进度：step {last['training/global_step']}/100\n\n"
        f"- 近 {len(recent)} 步平均：步时 {fmt((avg('timing_s/step') or 0) / 60, '.1f')} 分钟，"
        f"reward {fmt(avg('critic/rewards/mean'))}，grad {fmt(avg('actor/grad_norm'))}\n"
        f"- harness：mimocode {fmt(avg('harness/mimocode-agent/reward_mean'))} / "
        f"DSH {fmt(avg('harness/dsh-sdk/reward_mean'))}\n"
        f"- 被接收的组：Code {code} / Harbor {harbor}\n"
        f"- DSH 失败累计 {s.get('dsh_failures')}，ungradable {s.get('ungradable')}，沙箱 {s.get('sandboxes')}\n"
        f"- 预计剩余 {remaining:.0f} 小时"
    )


def tick(run: str, send: bool, auto_stop: bool) -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    try:
        snapshot = json.loads(remote(REMOTE % {"run": run}, python=True))
    except Exception as error:  # noqa: BLE001
        key, now = "ssh", time.time()
        if now - state.get("fired", {}).get(key, 0) > REPEAT_SECONDS:
            notify(f"## 看门狗：无法连接 GPU pod\n\n{error}", send)
            state.setdefault("fired", {})[key] = now
            STATE.write_text(json.dumps(state))
        log(f"ERROR {error}")
        return
    try:
        snapshot["sandboxes"] = int(remote(SANDBOXES % {"app": MODAL_APP}, python=False))
    except Exception as error:  # noqa: BLE001
        snapshot["sandboxes"] = None
        log(f"WARN sandbox count failed: {error}")
    alerts, facts = evaluate(snapshot, state)
    done = (snapshot.get("steps") or [{}])[-1].get("training/global_step") or 0
    now = time.time()
    fired = state.setdefault("fired", {})
    for key in list(fired):
        if key not in {a[0] for a in alerts} and key != "ssh":
            fired.pop(key)
    systemic = [a for a in alerts if a[2]]
    for key, text, _ in alerts:
        if now - fired.get(key, 0) > REPEAT_SECONDS:
            notify(
                f"## ⚠️ group-a-r1 看门狗报警\n\n{text}\n\n- step {done}/100，距上一步 {facts['idle_min']:.0f} 分钟",
                send,
            )
            fired[key] = now
    if systemic and auto_stop and not state.get("stopped"):
        result = remote(STOP % {"app": MODAL_APP}, python=False)
        state["stopped"] = now
        notify(
            f"## 🛑 已自动停止 group-a-r1\n\n原因：{'；'.join(a[1] for a in systemic)}\n\n结果：{result}。"
            "已回收本 run 的 Modal 沙箱，修复后从最近 checkpoint 恢复。",
            send,
        )
    if done and done % SUMMARY_EVERY == 0 and state.get("summary_step") != done:
        notify(summary(snapshot), send)
        state["summary_step"] = done
    state["dsh_failures_seen"] = snapshot.get("dsh_failures", 0)
    STATE.write_text(json.dumps(state))
    log(
        f"OK steps={done} idle={facts['idle_min']:.0f}m sandboxes={snapshot.get('sandboxes')} "
        f"dsh_fail={snapshot.get('dsh_failures')} "
        f"sess_fail={sum(snapshot.get('session_failures') or [])}/{sum(snapshot.get('session_totals') or [])} "
        f"gateway={snapshot.get('gateway_http')} alerts={[a[0] for a in alerts]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--send", action="store_true", help="send Feishu messages (default: log only)")
    parser.add_argument("--auto-stop", action="store_true")
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--run", default=RUN, help="run dir holding metrics.jsonl, training.log, exit-code.txt")
    args = parser.parse_args()
    while True:
        tick(args.run, args.send, args.auto_stop)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
