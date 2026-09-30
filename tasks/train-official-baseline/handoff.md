# TL;DR

**2026-10-01用户要求评估恢复服务器、比较模式/参数并写脚本注释。本轮未创建新Pod；平台goal仍paused。旧截止已过，历史launcher必须重新绑定身份/期限/守护。**

官方优先，当前更新 0；源码 a2ad9f6160b03ff2d47e59832bfb6b289f37c917。
4×RTX PRO6000 96GB；官方 colocate_async，actor TP4 / rollout TP2，9B SFT，64K。
原服务器 root@157.157.221.177:30134 在07:49 UTC已不可达；控制面Pod not_found、列表为空。regional卷72jdno5cuk仍在。截止 2026-09-30T12:51:43.743Z 不延长。

## 本轮交付物

- `docs/train-official-baseline/README.md` — 82行；官方方案、复现入口、测试与lint修复审计。
- `docs/train-official-baseline/design.md` — 23行；官方方案、复现入口、测试与lint修复审计。
- `docs/train-official-baseline/evidence/deployment-receipt.json` — 23行；原始身份/运行配置/部署或暂停证据。
- `docs/train-official-baseline/evidence/launch.sh` — 117行；原始身份/运行配置/部署或暂停证据。
- `docs/train-official-baseline/evidence/manifest.json` — 3720行；原始身份/运行配置/部署或暂停证据。
- `docs/train-official-baseline/evidence/operations-deployment.json` — 1行；原始身份/运行配置/部署或暂停证据。
- `docs/train-official-baseline/evidence/pause-local-services.json` — 17行；原始身份/运行配置/部署或暂停证据。
- `docs/train-official-baseline/evidence/pause-status.json` — 16行；原始身份/运行配置/部署或暂停证据。
- `docs/train-official-baseline/lint-baseline-repair.json` — 108行；官方方案、复现入口、测试与lint修复审计。
- `docs/train-official-baseline/operations/close_at_deadline.py` — 506行；固定截止清理/独立测试/真实进程监督。
- `docs/train-official-baseline/operations/test_close_at_deadline.py` — 157行；固定截止清理/独立测试/真实进程监督。
- `docs/train-official-baseline/operations/train_driver.py` — 70行；固定截止清理/独立测试/真实进程监督。
- `docs/train-official-baseline/prepare_baseline.py` — 351行；官方方案、复现入口、测试与lint修复审计。
- `docs/train-official-baseline/test_prepare_baseline.py` — 244行；官方方案、复现入口、测试与lint修复审计。
- `examples/tutorial/agent_loop_get_started/agent_loop_tutorial.ipynb` — 931行；仅修复官方历史lint/格式；GPU仍使用冻结原版。
- `examples/tutorial/ray/tutorial.ipynb` — 966行；仅修复官方历史lint/格式；GPU仍使用冻结原版。
- `recipes/arvo/env_actor.py` — 262行；仅修复官方历史lint/格式；GPU仍使用冻结原版。
- `recipes/design/agent_loop.py` — 1242行；仅修复官方历史lint/格式；GPU仍使用冻结原版。
- `recipes/design/env_actor.py` — 303行；仅修复官方历史lint/格式；GPU仍使用冻结原版。
- `recipes/general/agent_loop.py` — 1068行；仅修复官方历史lint/格式；GPU仍使用冻结原版。
- `recipes/general/env_actor.py` — 384行；仅修复官方历史lint/格式；GPU仍使用冻结原版。
- `recipes/general/general_agent/environment.py` — 717行；仅修复官方历史lint/格式；GPU仍使用冻结原版。
- `recipes/general/general_agent/k8s_sidecar.py` — 284行；仅修复官方历史lint/格式；GPU仍使用冻结原版。
- `tasks/lessons.md` — 5行；计划、经验或交接；当前暂停。
- `tasks/todo.md` — 22行；计划、经验或交接；当前暂停。
- `tasks/train-official-baseline/handoff.md` — 34行；计划、经验或交接；当前暂停。
- `tests/recipes/design/test_webdev_launcher_on_cpu.py` — 332行；仅修复官方历史lint/格式；GPU仍使用冻结原版。
- `verl/utils/length_penalty.py` — 358行；仅修复官方历史lint/格式；GPU仍使用冻结原版。

本地完整备份：`/Users/gumpm5/.local/share/xdan-train-performance-verl/pause-20260930/`；含纯官方部署archive、四profiles、命令、manifest及两分支Git bundle，约55MB，全量SHA与bundle verify通过。`manifest.json`标记实际保存的分支HEAD。

## 设计约束

主目录 main 不动；纯官方训练核心零补丁，DSH 暂缓。模型/uv/数据复用已验证资产，不使用旧 checkpoint。凭据不入仓库/argv。

## 已发现真实行为

官方 validator 只接受 colocate_async；原版 Code runner 要求至少两 harness，使用四原生 profiles。默认 logger console/tensorboard；wandb/file/rl_insight 已原生支持。扩展版本 init-only 成功不构成官方 fit 证据。

## 下一里程碑

- [x] 源码部署与服务器 CPU 配置预检（1810files+6links；07:42 UTC exit0）。
- [ ] 获取当前新实例状态/SSH，再核验云盘与uv资产。
- [ ] 预算清理脚本覆盖新 source/run。
- [ ] step1 更新/保存，恢复 step2/holdout，监控验收。

## 分支/部署状态

worktree-train-official-baseline，从官方 a2ad9f61 创建；af4164b7/f6f7e96b已push，暂停交接另有后续提交。DSH WT最新本地保存c3979814（含140 CPU测试证据），全库历史notebook/固定vendor门禁未过，未push。source/12配套运行文件已部署，CPU预检通过。新closer启动SSH命令失败，无法确认执行，不声称armed。本地guard32896/caffeinate32897已停止并独立ps确认均不存在；原Pod现not_found、Pod列表为空，远端guard当前不可查。更新清理守护后再启动 fit。

## 冷启动 checklist

先读本文件、tasks/todo.md、docs/train-official-baseline/design.md。核对当前时钟与固定截止；SSH 检查 GPU/guard/run 原始日志，再决定下一步。不要沿用旧状态声称训练已启动。

## 2026-10-01补充交付与结论

- `docs/train-official-baseline/script-comparison.md` — 144行；三列参数/路径、三模式、速度/质量与容量证据边界。
- `docs/train-official-baseline/launch-4gpu-colocate-annotated.sh` — 144行；历史launcher的就地解释；有效tokens不变。
- `docs/train-official-baseline/prepare_baseline.py` — 401行；只新增注释输出，官方源码/历史evidence保持不变。
- `docs/train-official-baseline/script-annotation-verification.json` — 11行；两条shell的tokens/bash验收，未产生GPU任务。
- `docs/train-official-baseline/restore-assessment-20261001.md` — 31行；同卷、同镜像、持久uv快照及增量wheel恢复计划。
- integration的`scripts/code/train-dsh-separate-async.sh` — 新增15行注释，本地commit f899fe96；保留历史untracked文件，历史全库门禁仍阻挡push。

余额查询约$2996，原standard4000GB区域卷仍在；4×RTXPRO6000合计$8.36/h、库存Low，区域四卡成交未保证。新Pod未启动。12小时候选GPU费用约$100.32，旧截止不续用。

官方四卡colocate采样阶段两TP2 replica，训练TP4；DSH 2+2稳态只有一个独立TP2 replica，训练TP2但可重叠。锁定源码should_switch_to_rollout直接False，不存在已实现的持续空闲借池策略。原版Code validator只接受colocate_async。

8项现有CPU测试通过；注释tokens不变/bash语法通过；官方全库Ruff0.12.2两项通过。当前实际官方GPU更新0，未形成速度或质量对照证据。恢复时先做原版闭环，随后固定算法/harness作拓扑比较。
