# Lessons

- 用户纠正优先级为官方基线优先：先固定官方完整源码与子模块，走原版入口证明更新/保存/恢复，再叠加 DSH 与 separate_async。不能只换外层入口仍运行已修改核心而称为官方基线。
- logger 支持、启用与实际收到本次训练数据须分别验收；W&B API 与 RL-Insight trace 要匹配新 run。
- 不重复重建已通过实际 CUDA/模型初始化的 uv 环境；优先消除训练路径差异。
- 三种模式须核对实际注册、Code validator 和生命周期；separate_async 的类注释称可切回采样，但锁定版本 should_switch_to_rollout() 实际返回 False，不能把 hook 当作已实现的空闲借卡策略。
- 两卡容量候选、四卡脚本支持、真实完整闭环通过是三个独立结论；不能用显存估算或 CPU compose 给出已跑通保证。
