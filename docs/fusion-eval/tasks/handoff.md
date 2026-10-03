# fusion-eval 交接（2026-10-03，由 xdan-verl-fusion 训练会话创建）

## 1. TL;DR

- 本 worktree 只做**评测**：在一台独立的**双卡评测 pod** 上，为融合线 A 组跑 SFT 基线和 RL checkpoint 的评测。
- 训练由 `xdan-verl-fusion` 会话负责（pod :30293，run `group-a-r1`，W&B `ga103`）。本会话**不碰训练 pod 上的任何进程**。
- 第一优先级：**在约 2026-10-03 13:00 UTC（训练 step 25）之前**拿到 SFT 在 TB2.1 上的 mean@8（两个 harness）。
- 评测口径统一：**96K 上下文**；主指标 TB2.1 mean@8，次指标 Code holdout100 mean@4；两个 harness（mimocode、DSH）都评。
- 评测 pod 的 SSH 地址由用户提供。拿到地址前，先完成第 5 节的代码准备。

## 2. 目标与评测口径（用户已定，不要改）

| 项 | 取值 |
|---|---|
| 主指标 | TB2.1（89 题）mean@8，温度 1.0，两个 harness 分别报告 |
| 次指标 | MiMo Code holdout100 mean@4，两个 harness |
| 训练目标 | 在 TB2.1 或 holdout100 中至少一项比 SFT 高 ≥ 3pt，另一项不退步（按 harness 分别判断） |
| 上下文 | 96K：MAXLEN 98304、RESPONSE_LENGTH 94208、SGLang context_length 98304、DSH context_window 98304；单轮输出上限 32768 |
| 待评对象 | SFT（`/workspace/models/MiMo-V2.6-Distill-Qwen-9B`），以及 `checkpoints/group-a-r1-milestones/step_{25,50,100}`（由训练 pod 上的 milestone_keeper 自动复制，带 `config.json` 才算完成） |
| 配对比较 | 每题保存逐次得分，用于和 SFT 做配对差与标准误（粗估 TB2.1 mean@8 的配对标准误约 2.4pt） |

排队顺序（双卡吞吐有限，每个 checkpoint 的完整矩阵约 13–19 小时）：
1. SFT × TB2.1 × {mimocode, DSH} mean@8（1,424 条轨迹）——赶在 step 25 之前
2. SFT × holdout100 × 两个 harness，mean@4
3. step 25 → step 50 → step 100，每个都先跑 TB2.1，再跑 holdout100
4. 空档：batch1 难度 pilot（200 题 × 4 次，SFT，mimocode），详见 `docs/xdan-verl-fusion/data-scaling-analysis.md`

## 3. 设计约束（铁律）

- **不碰训练 pod（:30293）上的任何进程**，也不改 `runs/group-a-r1*`、`checkpoints/group-a-r1*`（milestones 目录只读）。baseline 线的 pod（:10924）同样不碰。
- 评测 pod 必须和训练 pod 在**同一机房 EUR-IS-1**，并挂载同一个网络卷 `72jdno5cuk`。模型、数据、checkpoint 直接走 `/workspace`。
- Modal app 名用 `xdan-fusion-eval-a`（launch_eval_a.sh 已设置），清理时只按这个 app 过滤。
- 评测结果写到 `/workspace/xdan-verl-fusion/runs/eval-a-<tag>-<harness>[-<bench>]/`。训练会话的看门狗用 `runs/eval-a-sft-*/metrics.jsonl` 判断 SFT 基线是否完成，**不要改这个前缀**。
- 不改训练用的 `config/agent/mixed/*.yaml`（训练正在使用它们）。96K 评测另建 profile，比如 `config/agent/mixed/dsh-sdk-modal-96k.yaml`，以及对应的 `*-only-96k.yaml`。
- 密钥只放评测 pod 的 `/root/mimo-private`（0600），用 `scripts/provision_secrets.sh` 从 Mac 下发；**不要打印密钥，也不要写进仓库或日志**。
- 评测隔离：TB2.1 和 holdout100 本身就在训练黑名单里，不要拿它们做任何训练用途。
- 提交只在 `worktree-fusion-eval` 分支上。文档放 `docs/fusion-eval/`。push 前跑 `ruff check` 和 `ruff format --check`。commit 加 `Co-Authored-By` trailer。

## 4. 已知的真实行为（训练会话踩过的坑）

- **DSH 的 profile 必须和 mimocode 逐项对齐**：单轮 32768、run_timeout 4800、命令 timeout 300、温度 1.0。曾经有 4096 的单轮上限 bug，导致 48% 的会话被截断。
- **DSH 的 `max-tokens` 结束原因包含两种情况**：单轮输出撞上限，或上下文用满。要按最后一轮的 usage 区分（输出 ≥ 30000 才算撞单轮上限）。
- **上下文用满的会话仍然会判分**，因为沙箱里已有的修改照样算数。64K 下：mimocode 有 24.5% 的会话用满上下文，这些会话平均 0.48 分；DSH 是 36%，平均 0.33 分。
- **官方 Code 镜像带有任务之后的 git 历史**，profile 里必须设 `git_leak_prevention: strip`（mixed profile 已设置）。
- **DSH 需要网关**。网关跑在**评测 pod 自己身上**，用 `docs/xdan-verl-fusion/ops/start_dsh_services.sh`，配 RunPod HTTP 端口代理，监听 0.0.0.0。上游超时要 ≥ 3600（默认就是）。在 RunPod 后台给评测 pod 开放一个 HTTP 端口（例如 8000）。
- **pkill/pgrep 一律用 `^` 锚定**，否则会匹配到自己的 ssh 命令。
- **W&B run id 不能复用**，每次评测都换一个新的。
- `--cfg job` 不会实例化 dataclass，新 launcher 要做一次真实启动的冒烟测试。

## 5. 待办（可勾选）

- [ ] 写双卡评测 launcher：`docs/fusion-eval/ops/launch_eval_2gpu.sh`。以 `docs/xdan-verl-fusion/ops/launch_eval_a.sh`（4 卡版）为基础，改为 TRAIN_NGPUS_PER_NODE=2、ACTOR_TP=2、ROLLOUT_TP=1（2 个副本）、MEGATRON_OFFLOAD=True，长度改为 96K，并支持 `BENCH=tb21|holdout`（分别用网络卷上的 `/workspace/xdan-verl-fusion/data/eval-a/tb21.parquet` 和 `.../eval-a/code-holdout100.parquet`）以及 VAL_N。
- [ ] 建 96K 评测 profile（DSH context_window 98304；mimocode 不需要单独设置），以及 tb21/holdout 的 only-spec
- [ ] 评测 pod 冷启动（见第 7 节）：环境、密钥、Ray、DSH 网关（带 PUBLIC_ORIGIN）
- [ ] 冒烟：SFT × TB2.1 × mimocode，取 4 题、VAL_N=2，测吞吐（token/s、每小时轨迹数）
- [ ] 正式队列：按第 2 节的顺序，写 `docs/fusion-eval/ops/eval_queue.sh`，任何一项失败都只记录、继续下一项
- [ ] 每次评测输出 `summary.json`：mean@k、各题逐次得分、各 harness 和 benchmark 的结束原因分布
- [ ] 配对比较脚本：RL 相对 SFT 的逐题配对差、标准误、按 harness 拆分
- [ ] 结果和交接同步回 `docs/fusion-eval/tasks/handoff.md`

## 6. 分支和部署状态

- 分支 `worktree-fusion-eval`，基于 `worktree-xdan-verl-fusion`，起点就是本文件所在的提交。
- 训练 pod 上的代码目录是 `/workspace/xdan-verl-fusion/source-5d234637`。评测要用到的新 profile 需要按同样方式部署成新的 `source-<commit>` 目录（git archive → 解包 → 从 `source-6c702c5b` 复制 third_party），**不要覆盖训练正在用的目录**。
- 评测 pod：待用户提供。

## 7. 冷启动 checklist

1. 读本文件，再读 `docs/xdan-verl-fusion/tasks/lessons.md` 和 `docs/xdan-verl-fusion/data-scaling-analysis.md`。
2. 看 `docs/xdan-verl-fusion/ops/` 下的 `launch_eval_a.sh`、`eval_a_queue.sh`、`start_ray_fusion.sh`、`start_dsh_services.sh`、`fusion-runtime.env`。
3. 评测 pod 的环境：
   - uv 环境的离线 bundle 在网络卷上：`/workspace/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/bundles/`；
   - 恢复步骤见 baseline 线文档 `.codex/worktrees/train-official-baseline/docs/train-official-baseline/uv-runpod-complete-plan-20261001.md`，只读，不要改那个 worktree；
   - 恢复后要有 `/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/activate.sh`。
4. 密钥：在 Mac 上执行 `bash <网络卷 scripts>/provision_secrets.sh <host> <port>`，具体用法看脚本头注释；DSH 的 routes 由 start_dsh_services.sh 生成。
5. 运行状态在训练会话的看门狗里可以看到（飞书推送），评测结果出来后由训练会话负责解读和推送。
