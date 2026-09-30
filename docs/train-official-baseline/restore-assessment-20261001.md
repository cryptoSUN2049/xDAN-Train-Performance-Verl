# 恢复服务器的具体方案（2026-10-01 新加坡时间）

当前仅完成资源查询、脚本注释和恢复方案；未创建新 Pod。旧 Pod 已删除，不能通过 start 恢复原实例。当前账户余额约 $2996；regional volume `72jdno5cuk`（EUR-IS-1，STANDARD，4000GB）仍存在，账户 Pod 列表为空。

最新 capacity 查询：4×RTX PRO6000 Blackwell Server Edition 96GB，Secure Cloud，报价合计 $8.36/h，stock Low，CUDA13.2 显示 AVAILABLE。此前查询显示 unavailable，说明库存有变化。区域列表也显示 EUR-IS-1 此型号 Low，但 capacity 不提供 DC 过滤，**不保证本区域同机四卡一定能成交**；创建前再核对。

## 拟恢复配置

- 四卡同机，绑定原 regional volume 到 `/workspace`；不新购网盘、不迁移旧 checkpoint。
- 沿用 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` 镜像，保持系统栈；项目 uv 从现有 archive 恢复。
- 容器临时盘保持上次500GB，承载本项目 `/opt` 环境与活跃 checkpoint；关机前备份回云盘。
- 官方 a2ad9f61 + 固定子模块，README SFT9B，65536上下文，colocate_async，训练TP4 / rolloutTP2。
- 原生四 harness 与 Modal；先更新1 → 保存CP1 → 恢复更新2 → holdout，不计作DSH完整覆盖。
- 新 run/W&B ID、全新 supervisor/cleanup 所有权；不能原样运行历史20260930 launcher。
- 延续12小时上限，**从新实例实际启动时重新绑定一个经确认的截止时间**；GPU最高约$100.32，另有既存存储和Modal成本。旧截止已过，不复用旧guard。

## 恢复顺序与验收

1. 实例创建成功后先绑定新 Pod 身份、费用及独立本地/远端停机守护。
2. 读取真实 GPU/driver/cgroup 与挂载；检查源码、模型、parquet、快照、native wheel 的完整SHA。
3. 恢复5.022GB archive 到相同 `/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654`。候选snapshot包含273包，尚未通过新Pod冷恢复验收。
4. 复用持久的 Apex wheel（SHA902f0854…）与cachetools增量锁，核对275包freeze；不要重新编译全部native，也不宣称snapshot已覆盖275包。
5. 在新Pod验证CUDA数值/四卡通信、Ray真实worker认证、监控后端及官方resolved config。配置中的64K容量仍需要实际GPU训练证明。
6. 官方原版入口真实fit，验收非零有效更新、完整CP、恢复后更新及holdout；API/文件/Insight对应本次run。
7. 更新完整环境快照与runbook，备份checkpoint/日志再关GPU；之后回到DSH/2+2阶段。

## 脚本整理

本目录集中存放 `script-comparison.md`、`launch-4gpu-colocate-annotated.sh`、`prepare_baseline.py` 及历史evidence。DSH源入口仍在 `train-p0-integration/scripts/code/train-dsh-separate-async.sh`，本轮也加入参数解释。两个worktree均在同一仓库内，核心实现不同，不直接互换入口。历史evidence不覆盖，源码archive仍保持纯官方字节。

注释改动已验证shell有效tokens完全一致、bash语法通过；8项现有官方CPU配置/源码identity测试通过。该验证不产生GPU任务，不代表真实训练已跑通。
