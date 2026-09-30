# 官方基线优先

- [x] 核实官方仓库、README、Code 入口、固定 SHA。
- [x] 建立独立 worktree 与设计；用户明确授权官方优先。
- [x] 准备四卡/64K/四原生 harness 配置，CPU compose 与官方 validator 通过。
- [ ] 部署纯官方源码，逐文件校验；保持健康 uv 环境。
- [ ] 核对 W&B/RL-Insight 配置与截止清理守护。
- [ ] 执行服务器真实配置预检和官方训练。
- [ ] 完成 step1 更新与完整 checkpoint。
- [ ] 从本次 checkpoint 恢复，完成 step2 与 holdout。
- [ ] 汇总客观日志/API/监控证据、commit；满足全库 Ruff 后 push。
- [ ] 返回 DSH 与 2+2 分池的后续验收。

## Review

CPU 配置通过不等于训练通过。当前真实官方更新数 0；禁止将已通过的扩展版本初始化算作官方闭环。
