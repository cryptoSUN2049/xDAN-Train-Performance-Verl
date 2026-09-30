# 官方 MiMo Code 基线：源码、脚本与监控

## 官方来源

- [XiaomiMiMo/verl README](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/README.md)：Code 训练、数据、SFT 9B、任务镜像和两个子模块。
- [官方入口 scripts/code/train.sh](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/scripts/code/train.sh)。
- [官方环境变量 scripts/code/env.example](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/scripts/code/env.example)。
- [底层 recipes/code/run_train.sh](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/recipes/code/run_train.sh)：配置解析、集群预检、原版 validator 后进入 `python3 -m verl.trainer.main_ppo`。
- [训练镜像说明 docker/README.md](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/docker/README.md)。README 中 `xiaomimimo/mimo-v2.6-rl-oss` 按任务提供执行镜像，不能直接当统一训练环境镜像。
- [官方 Tracking 实现](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/verl/utils/tracking.py) 与 [RL-Insight 文档](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/docs/advance/rl_insight.md)。

## 当前真实状态

用户已明确暂停服务（2026-09-30 07:57 UTC）。本地旧guard/caffeinate已停止并确认进程不存在；平台goal=paused。没有创建新GPU实例。下一轮必须等用户明确恢复。

2026-09-30 07:42 UTC，服务器执行本次官方 launcher 的 `CPU_CONFIG_PREFLIGHT=1` 成功返回 0，stdout 明确报告 `preflight passed`。这包含实际 Pod Python 的配置解析和原版 validator，**不包含 GPU 初始化或 fit**。

随后 SSH 在握手阶段断开。07:49–07:52 UTC，Runpod MCP 与已认证 runpodctl 对 `oeab4wv7wpnc8x` 查询均返回 `not_found`，Pod 列表为空；regional volume `72jdno5cuk` 仍在 EUR-IS-1。删除/结束原因没有证据，不能归因为训练故障，也不声称新截止清理脚本已启动。当前官方真实更新数 **0**。

源码已在旧实例 `/workspace/train-p0-dsh-integration/source-a2ad9f61` 部署，1810 regular files + 6 symlinks 全量验证。官方源码、已提交配套脚本、临时环境、云盘归档是四个独立资产，身份见 `evidence/`。两个 archive 打包格式不同，SHA 不同；实际部署使用 `deployment-receipt.json` 中的 `eadb21af...` 未压缩 archive。

## 固定配置及接口

| 项目 | 本次设置 |
|---|---|
| 模型 | README 的 MiMo-V2.6-Distill-Qwen-9B SFT；从头开始本轮 RL |
| GPU | 4×RTX PRO6000 Blackwell 96GB |
| 拓扑 | 官方 colocate_async 四卡共享；actor TP4/CP1；rollout TP2 |
| 上下文 | 65536；prompt4096 + response61440；显式 SGLang context_length65536 |
| 采样/批量 | N4；global batch1 / mini batch1 / micro batch1；static batch |
| Harness | 原版 mini-mimocode、mini-bash、mini-claude-code、mini-codex；step-hash |
| 环境 | 原生 Modal 后端，固定任务镜像和评分器；没有接入 DSH |
| 首段 | 真实 step1，SAVE_FREQ1；保存 model/HF/optimizer/extra/data |
| 恢复段 | 从本轮 CP1 恢复至 step2；holdout1，独立 run |
| 日志 | 官方 console/tensorboard/file/wandb/rl_insight |
| 预算 | 原硬截止 2026-09-30T12:51:43.743Z 不自动延长 |

调整只是资源、上下文、任务环境、监控与保存设置。训练核心始终为官方 a2ad9f61 和官方固定子模块。为了满足用户规定的全库 push lint gate，worktree 中另有格式修复；Python AST/Notebook outputs 不变，且该格式修复没有进入已部署官方源码。

## W&B 和 RL-Insight

原版已提供两个 logger，Code 脚本默认只启用 console/tensorboard。配套 launcher 显式启用两者：

```bash
trainer.logger='[console,tensorboard,file,wandb,rl_insight]'
export RL_INSIGHT_SERVER_URL=http://127.0.0.1:18080
export VERL_RL_INSIGHT_ENABLE=1
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
```

W&B 通过 SDK 写入，API 用于独立读取 run/history，不能把“能调用 API”当“已有训练指标”。认证采用 NETRC 私有路径；Modal 同样传配置文件路径。Ray runtime_env 必须转发这些路径、run 元数据和 Insight URL/enable，API key 不进 Hydra argv/源码/日志。

平台收到真实 step 后，检查新 run a9off001/a9off002 的 history 和参数，文件指标与 Insight 标量一致，trace 关联本次 Ray job/actor。当前只完成配置，不声称收到官方训练数据。

## 重建与执行

本机生成原字节配置、原生 profiles、源码身份和最小差异 manifest：

```bash
python3 docs/train-official-baseline/prepare_baseline.py \
  --output-dir /tmp/train-official-baseline-NEW \
  --vendor-root /Users/gumpm5/Documents/Code/xDAN-Train-Performance-Verl/.Codex/worktrees/train-p0-integration
```

部署文件和实际配置见 `evidence/launch.sh`。服务器 CPU 预检与生产入口：

```bash
CPU_CONFIG_PREFLIGHT=1 bash /workspace/train-p0-dsh-integration/runs/official-code-4gpu-64k-r1-20260930/launch.sh fresh
# 确认当前实例、GPU、uv、Ray、认证、监控和固定截止守护后，再运行 supervisor：
/opt/env_infra/rtx6000/xdan-train-performance-verl-uv-6e6cc6b2978a1654/venv/bin/python3 \
  /workspace/train-p0-dsh-integration/runs/official-code-4gpu-64k-r1-20260930/train_driver.py --phase fresh
```

`resume` 必须以本轮 CP1 的完整文件/身份验收为前提，且从 `resume-step2/train_driver.py --phase resume` 启动。保存成功、恢复载入成功与恢复后真实更新分别验收。不可复用失败 run 的 claim/log，重试使用新目录并同步调整严格归属清理配置。

新 Pod 要先挂同 regional volume，检查 `/workspace/env_infra/rtx6000/`。旧 uv 在 `/opt` 临时层，已有持久快照候选只覆盖273包，后来增加 cachetools/Apex 到275；冷恢复和增量 wheel/hash 仍需核对。禁止把候选快照称为已通过完整重建，不盲目重复全部 native 编译。

## 本机验证与提交

21 CPU tests passed，包含 fresh/resume Hydra compose、官方 validator、固定截止归属检查。官方 `.pre-commit-config.yaml` 固定 Ruff0.12.2；全库 `ruff check .` 与 `ruff format --check .` 均通过，725文件格式检查通过。

已推送 `worktree-train-official-baseline`：`af4164b7`（门禁修复）、`f6f7e96b`（复现方案）。真实 GPU 更新/checkpoint/恢复/holdout 尚待验收；之后再接回 DSH/2+2/较大数据量的主线。
