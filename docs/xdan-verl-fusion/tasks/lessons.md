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
- **换启动入口后，必须把 resolved config 和已验收的配置逐项 diff。** 第一次启动经过了 `train-dsh-minimal.sh`，没有察觉它带着 2 卡场景的默认值：prefill 4096、mamba 缓存 16、rollout 显存 0.6。结果吞吐只有 23 token/s，跑了 2 小时只完成 2 题。改回官方 Code 入口后吞吐约 1,050 token/s，单副本。规则：任何正式 run 开跑前，都要和最近一次通过验收的同类 run 做 resolved_config diff，**逐条说明每处差异是有意为之的**。脚本见 `ops/` 下的 diff 片段，模板参考这次对比 r2 的做法。
- **W&B 的 run id 不能复用。** `WANDB_RESUME=never` 时，被弃用的旧 run id 也会让新 run 启动失败。每次重新开跑都要换一个 id。
- **混合 harness 的各 profile 也要逐项 diff，不能只比 resolved config。** `train.sh` 只会把官方 four-whitebox 组合的单轮输出上限改写为 `HARNESS_TURN_MAX_TOKENS`（32768）。自写的 mixed profile 不在改写范围内：mimocode 的 profile 是从 r2 复制的，自带 32768；DSH 的 profile 沿用了 minimal 脚本的 4096。结果 steps 1–7 中，DSH 有 155/322 个会话因为单轮思考被截断（`turn/end reason=max-tokens`）而判为 LimitsExceeded，平均得分 0.13；正常完成的 DSH 会话得分 0.63，和 mimocode 的 0.60 持平。这个 4096 上限白白浪费了一半 DSH 采样，还会给长思考打负 advantage。规则：同一个 mix 里的各 harness，`model_kwargs`（max_tokens、timeout、temperature）必须一致，开跑前逐项 diff。看到「某个 harness 明显更弱」时，先按结束原因（finish/turn-end reason）拆分统计，再下结论。
- **自匹配不只出现在 pod 上。** 在 Mac 上执行 `ps | grep "[w]atchdog.py" | xargs kill` 时，同一条 zsh 命令行里也含有 `watchdog.py`，结果把自己的 shell 杀了，排在后面的 commit 没有执行。规则：按进程名 kill 一律用 `pgrep -f "^<完整命令前缀>"`，并且单独执行，不要和其他步骤串在同一条命令里。
- **DSH 的 `max-tokens` 结束原因包含两种情况：单轮输出撞上限，以及 64K 上下文用满。** 只能按最后一轮的 usage 区分（输出 ≥ 30000 算撞上限，其余算上下文用满）。之前的看门狗没有区分，在 ga103 上发了一次误报：撞上限 0 次，上下文用满 135 次。
- **调整超时时，相关的有效期要一起改。** DSH 网关的会话路由 TTL = `run_timeout + 60`，从会话注册时就开始计时；但沙箱启动和 payload 注入也要花时间，而 run_timeout 是 DSH 进程启动后才开始计时。把 run_timeout 从 3600 调到 4800 之后，长会话在跑到第 4465–4789 秒时，路由先过期了，返回 HTTP 401（fusion-eval 的 SFT×TB2.1×DSH 出现了 5 次以上）。已把余量改为 900 秒。规则：改任何超时，都要列出所有依赖它的 TTL 和超时，逐一检查。
