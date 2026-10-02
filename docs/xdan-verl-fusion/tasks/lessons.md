# xdan-verl-fusion 踩坑与规则

## 共享网络卷清理（2026-10-02）

- 规则：清理 `/workspace/train-p0-dsh-integration/checkpoints` 或 `runs` 之前，必须先发消息给 baseline 会话（`xdan-train-performance-verl` baseline 线）协调，不能自行删除。
- 原因：baseline 的 GPU 队列会在 fresh 和 resume 之间依赖这些 checkpoint，"每个 run 只留最新一份"的清理一旦落在两步之间，resume 就会失败。baseline 有自己的保留策略 A（`prune_checkpoints.py`），由它自己执行。
- 做法：融合线只清理自己的 `/workspace/xdan-verl-fusion/`；其他目录哪怕用户已经授权大清理，也要先发消息协调，等对方确认后再删。

## 验证

- `main_ppo --cfg job` 只证明 Hydra 能把配置组装起来，不会实例化 dataclass。像上游删掉的字段（如 `grad_offload`）这类问题，要等到 worker 启动才会暴露。新 launcher 要加跑 `ops/dataclass_check.py`，或者对 resolved config 实际构造一次 dataclass。
- 判断失败是不是融合引入的：在 pristine mimo-oss 上跑同一个检查，只有新增的失败才算融合问题。

## Launcher

- `runtime.env` 把 PYTHONPATH 固定在 `source-a2ad9f61`。启动 Ray 前必须先 source `fusion-runtime.env`。
- recipe 的 run 脚本已经会把 PYTHONPATH 转发进 Ray runtime_env；launcher 再转发一次会让 Hydra 报重复键。
- `train-dsh-minimal.sh` 强制要求 DSH gateway，非 DSH 的 harness 要走 `scripts/code/train.sh`。

## 正式训练（group-a-r1，2026-10-02）

- **DSH 网关上游超时要不小于模型请求超时。** 网关代理对上游写死了 300 秒读超时（心跳只保护公网那一侧）。rollout 引擎满载时，DSH 单轮排队时间会超过 300 秒，结果流被截断，报 `STREAM_CLOSED: SSE stream ended without [DONE]`。4 次失败的耗时都在 300.3 到 300.7 秒之间。已修复（`6cadc2c6`）：新增 `--upstream-timeout`，默认 3600 秒，和 `MODEL_REQUEST_TIMEOUT` 一致。冒烟测试负载低，所以没暴露这个问题；任何超时类参数都要在接近满载的情况下验证。
- **官方 Code 镜像保留了任务 base 之后的 git 提交。** 不在运行时 strip 的话，MiMo-Agent 会直接拒绝这类镜像（抽查 4 个 holdout 镜像中有 3 个）。在混合配置里设 `git_leak_prevention: strip`；只有 Code 行生效，Harbor 行会忽略。
- **混合数据集共用一份 harness yaml。** yaml 里只适用于 Code 的开关（反作弊清理、git strip）要在 `HarborEnvironment` 里强制关闭，否则会改坏 Harbor 任务。
- **选题流水线的 shell 嵌套引号会静默产出空文件。** 结果 stage1 的排除清单为空，batch1 混入了 184 道重复题。现在用 heredoc 写 Python，并在生成后校验行数下限。
- **`pkill -f <pattern>` 会匹配到自己的 ssh 命令行。** 用 `pgrep -f "^<完整命令前缀>"` 锚定后再 kill。
