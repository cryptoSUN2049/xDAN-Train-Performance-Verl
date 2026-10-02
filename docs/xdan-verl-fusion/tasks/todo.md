# xdan-verl-fusion 打通计划（2026-10-02）

目标：在 MiMo fork 的融合分支（`xdan/fusion-a9f2985`）上，把**任务 × harness** 正交地接进来，并用真实 GPU 训练验证整条链路。任务包括 MiMo 五域和 Harbor；harness 包括 MiMo 白盒 harness 和 DSH。

## 环境

- GPU：`157.157.221.177:30392`，4×RTX PRO 6000，专供融合线使用；网络卷 `72jdno5cuk`（EUR-IS-1）。
- venv：从网络卷快照恢复到 `/opt/env_infra/...-6e6cc6b2978a1654`，275 包已校验。
- 源码目录：`/workspace/xdan-verl-fusion/source-<commit>`，一个 commit 一个目录，不覆盖。
- 运维脚本：`/workspace/xdan-verl-fusion/ops/`，包括 `fusion-runtime.env`、`start_ray_fusion.sh`、`launch_music_fusion.sh`、`start_dsh_services.sh`、`launch_dsh_fusion.sh`。
- 注意：`runtime.env` 把 PYTHONPATH 固定在 `source-a2ad9f61`。融合线必须先 source `fusion-runtime.env`，再启动 Ray；PYTHONPATH 由 recipe 的 run 脚本转发给 Ray，launcher 不要再重复转发。

## 阶段 A：融合分支 GPU 冒烟（Music，对照 m1）

- [x] CPU：导入预检、dataclass 实例化检查
- [x] 修复：`grad_offload` 已被上游删除，recipes 里同步去掉（`877f54d3`）
- [ ] fresh：1 步，跑完保存 checkpoint
- [ ] resume：从 step1 恢复，跑到 step2
- [ ] 与 m1 对比：reward、耗时、显存、DAPO 过滤

## 阶段 B：M1b + DSH × mimocode（Code train8，Modal）

- [x] 挑入 5 个 P0 verl 修复：`0bfd0630`、`f0f3fc3f`、`2886da5c`、`52ea0d4e`、`5b0a83f0`
  - `ced8e69d` 跳过，它依赖未挑入的 `0fab9666`（有界采样器）
- [x] DSH runtime 移植（`01a2ad9c`），recipes 测试 354 个全部通过
- [x] CPU preflight 加 dataclass 检查通过
- [ ] GPU：proxy + tunnel → 2 步混合 harness 训练，每组 2 条 DSH + 2 条 mimocode，按 harness 计算 advantage
- [ ] 核对：两种 harness 都产出了轨迹和 reward；参数确实更新；Modal sandbox 全部回收

## 阶段 C：Harbor 任务接入（正交）

- [ ] 设计：`HarborEnvironment(DatasetEnvironment)`，注册到 `DATASET_REGISTRY`；reward 用 Harbor 自带的 verifier
- [ ] 数据转换：Harbor 的 `task.toml`、`instruction.md` 转成 MiMo 数据行
- [ ] 镜像策略：mimocode 和 DSH 分别怎么准备镜像
- [ ] 失败即拒收：评分阶段出现 infra 错误时丢弃该轨迹，不计为 0 分
- [ ] GPU：Harbor 小子集 × {DSH, mimocode}，跑 2 步

## 运行时发现

- `--cfg job` 只检查配置能否组装，不会实例化 dataclass。以后每个新 launcher 都要额外做一次 dataclass 实例化检查。
