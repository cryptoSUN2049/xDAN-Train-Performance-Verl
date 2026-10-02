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
