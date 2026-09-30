> 状态：2026-10-01用户要求评估恢复及模式/脚本比较；goal仍paused。资源查询与文档注释已完成，没有创建新GPU Pod；旧guard/截止不能复用。

# 官方基线优先

- [x] 核实官方仓库、README、Code 入口、固定 SHA。
- [x] 建立独立 worktree 与设计；用户明确授权官方优先。
- [x] 准备四卡/64K/四原生 harness 配置，CPU compose 与官方 validator 通过。
- [x] 部署纯官方源码，逐文件校验；保持健康 uv 环境。
- [ ] 核对 W&B/RL-Insight 配置与截止清理守护。
- [x] 执行服务器真实配置预检（07:42 UTC exit0；不含 GPU fit）。
- [ ] 恢复四卡实例：07:49 UTC 控制面 not_found、Pod列表为空；持久卷仍在。
- [ ] 启动官方训练（资源恢复与新清理守护验收后）。
- [ ] 完成 step1 更新与完整 checkpoint。
- [ ] 从本次 checkpoint 恢复，完成 step2 与 holdout。
- [ ] 汇总客观日志/API/监控证据、commit；满足全库 Ruff 后 push。
- [ ] 返回 DSH 与 2+2 分池的后续验收。

## Review

CPU 配置通过不等于训练通过。当前真实官方更新数 0；禁止将已通过的扩展版本初始化算作官方闭环。复现代码已 commit/push 至 f6f7e96b；21 CPU tests/full Ruff0.12.2 gates通过。原四卡实例消失原因未知，暂时无法继续 GPU 工作。

暂停收尾：本轮官方方案/原始身份/本地停止回执/README/handoff已落盘。DSH扩展与实际GPU证据另存c3979814，未部署后续retry，13个历史untracked文件保留不动。恢复必须重新核对资源和预算，不能自动续上旧deadline。

## 2026-10-01恢复评估与注释

- [x] 查询余额、regional卷及四卡库存；未创建资源。
- [x] 官方/DSH参数就地注释；bash语法、有效tokens不变及8项现有CPU测试通过。
- [x] 三列脚本对照、三种模式边界、速度/容量/质量分析落盘。
- [x] 复核should_switch_to_rollout=False，纠正自动借池推断并写入lessons。
- [x] 写明冷恢复uv273包快照与Apex/cachetools增量275包的证据边界。
- [x] DSH注释独立本地commit f899fe96；仍因全库历史门禁未过而不push。
- [x] 官方本轮注释/文档与全库Ruff门禁完成；随本次提交保存，push结果须核对远程SHA。

Review：两个配置都使用四卡，但采样池大小不同；没有对照测速。恢复后的真实fit、checkpoint、恢复后更新与监控验收仍为下一里程碑。
