# TL;DR

**用户已于2026-09-30 07:57 UTC明确暂停服务。平台goal状态paused；不恢复GPU/训练/付费资源，等用户明确恢复。**

官方优先，当前更新 0；源码 a2ad9f6160b03ff2d47e59832bfb6b289f37c917。
4×RTX PRO6000 96GB；官方 colocate_async，actor TP4 / rollout TP2，9B SFT，64K。
原服务器 root@157.157.221.177:30134 在07:49 UTC已不可达；控制面Pod not_found、列表为空。regional卷72jdno5cuk仍在。截止 2026-09-30T12:51:43.743Z 不延长。

## 本轮交付物

docs/train-official-baseline/design.md：方案与验收；prepare_baseline.py / test_prepare_baseline.py：可复现准备器与 CPU 核验；tasks/todo.md 与 lessons.md：进展和优先级纠正。

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
