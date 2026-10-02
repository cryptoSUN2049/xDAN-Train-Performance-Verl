# xdan-verl-fusion 打通计划（2026-10-02）

目标：在 MiMo fork 的融合分支 `xdan/fusion-a9f2985` 上，按**任务 × harness**正交接入，并用真实 GPU 训练把整条链路跑通。
- 任务：MiMo 五域、Harbor
- harness：MiMo 白盒、DSH

## 环境

- GPU 机器：`157.157.221.177:30392`，4×RTX PRO 6000，只给融合线用。网络卷 `72jdno5cuk`（EUR-IS-1）。
- venv：从卷快照恢复到 `/opt/env_infra/...-6e6cc6b2978a1654`，275 个包已核对。
- 源码：`/workspace/xdan-verl-fusion/source-<commit>`，每个 commit 一个目录，不覆盖旧的。子模块从 `source-6c702c5b/third_party` 复制。
- 运维脚本放在 `/workspace/xdan-verl-fusion/ops/`：
  - `fusion-runtime.env`：SOURCE/PYTHONPATH 覆盖层，当前指向 `source-beb7ad42`
  - `start_ray_fusion.sh`、`start_dsh_services.sh`（proxy:8766 + cloudflared）
  - 启动脚本：`launch_{music,dsh,harbor,harbor_dsh}_fusion.sh`
  - 队列：`gpu_queue_fusion.sh`、`gpu_queue2_fusion.sh`
- DSH 运行时 payload：`/workspace/xdan-verl-fusion/payloads/dsh-runtime-0.1.3a2-linux-x86_64.tar.gz`
  - sha256 `e118df1d…`，120MB
- 注意：`runtime.env` 把 PYTHONPATH 钉在 `source-a2ad9f61`，所以融合线必须先 source `fusion-runtime.env` 再起 Ray。PYTHONPATH 由 recipe 的 run 脚本转发，launcher 里不要重复转发。

## A. 融合分支 GPU 冒烟（Music，对照 m1）✅

- [x] CPU 导入预检；dataclass 实例化检查
- [x] 修复 `grad_offload`：上游 #7544 删除了它，recipes 同步删除（`877f54d3`）
- [x] fresh step1：reward 0.585，grad 0.349，显存 39.7GB，用时 1137s（m1 为 0.244 / 0.392 / 44.1GB / 820s）
- [x] resume step1→2：模型和 optimizer 都已加载；reward 0.427，grad 0.294，显存 54.4GB，用时 1140s
- [ ] 跟进：融合分支每步比 m1 慢约 39%，主要慢在 gen（683s vs 508s），原因待查

## B. M1b + DSH 接入

- [x] 挑入 5 个 P0 verl 修复（`ced8e69d` 依赖 `0fab9666`，跳过）
- [x] 移植 DSH runtime（`01a2ad9c`）；CPU 预检和 dataclass 检查通过
- [x] DSH 运行时 payload 注入（`4faed1ec`）：任务镜像里不必再预装 DSH；在 Harbor 镜像上 4/4 启动通过
- [ ] GPU：Code train8 × {DSH + mimocode}，paired-subgroup，按 harness 分组算 advantage，跑 2 步（队列 1 第三段）

## C. Harbor 任务接入（正交）

- [x] `HarborEnvironment`（`dataset_type: harbor`），复刻 Harbor verifier 的约定；失败即拒收（`beb7ad42`）
- [x] 数据转换（只支持带预构建镜像的任务）；oracle 检查 4/4：未改动时 0 分，跑参考答案后 1 分
- [ ] GPU：Harbor × mimocode，跑 2 步（队列 1 第二段，正在跑）
- [ ] GPU：Harbor × {DSH + mimocode}，跑 2 步（队列 2）
- [ ] 正式训练要用不在 TB2.1 里的任务。stage1 的 500 个任务需要从 Dockerfile 构建镜像（Modal `from_dockerfile` → `im-` id），这一步还没做

## 运行时发现

- `--cfg job` 只组装配置，不实例化 dataclass；新 launcher 一定要加跑 dataclass 检查。
- `train-dsh-minimal.sh` 强制要求 DSH gateway；非 DSH 的 harness 走官方 `scripts/code/train.sh`。
- TB2.1 是对方的评测基准，只用来验证链路，checkpoint 不保留。
