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
- [x] GPU：Code train8 × {DSH + mimocode}，paired-subgroup，按 harness 分组计算 advantage，跑 2 步（d1，rc=0）。step1 全对，grad 为 0；step2 DSH 0.5、mimocode 0.0，grad 1.04

## C. Harbor 任务接入（正交）

- [x] `HarborEnvironment`（`dataset_type: harbor`），复刻 Harbor verifier 的约定；失败即拒收（`beb7ad42`）
- [x] 数据转换（只支持带预构建镜像的任务）；oracle 检查 4/4：未改动时 0 分，跑参考答案后 1 分
- [x] GPU：Harbor × mimocode，跑 2 步（h1，rc=0）：reward 0.50→0.75，grad 1.27 / 1.01；checkpoint 已丢弃（TB2.1）
- [x] GPU：Harbor × {DSH + mimocode}，跑 2 步（hd1，rc=0）：DSH payload 注入 8 次；step2 DSH 0.75 / mimocode 0.25，grad 0.29；checkpoint 已丢弃
- [x] stage1（HF `gump2049/xDAN-Harbor-Stage1-Tasks` 的 ladderA-v1 切片）：500+8 的镜像全部构建完成（Modal 缓存命中）；oracle 试点 20/20 通过（`002e8c97`、`717d9acf`）
  - 数据：`/workspace/xdan-verl-fusion/data/harbor-stage1/{train,validation}.parquet`；镜像映射在 `image-map-*.json`
- [ ] 全量池 `gump2049/xDAN-Harbor-Stage1-Tasks-Full`（15,406 题，网盘在 `data-eval-set-v1/repo`）：已转换，但还没审计。计划先抽 2,000 题做构建和 oracle 审计，**等用户确认批量和 Modal 预算**
- [ ] 评测隔离机制：prepare_data 内置黑名单（TB2.1、SWE-bench Verified、eval-set-v1、各 holdout），命中直接报错

## D. 已处理数据的复用（盘点：2026-10-02）

- [ ] Code 2698 训练集里含有 holdout100 → 切出 2598 条干净的 train，作为任务 × harness 的主力数据。用 DSH payload，不需要建镜像
- [ ] train8 / minimal：GHCR 上的单任务 DSH 镜像换回官方镜像，验证 payload 可以完全替代它们
- [ ] 对方 r4/r6/r7/r8 中新增的 47 题：build_images + oracle
- Cyber / General / Webdev 的 verifier 写死在 AgentLoop 里，暂不进 harness 矩阵；Music 是单轮任务，不在矩阵内

## 运行时发现

- `--cfg job` 只组装配置，不实例化 dataclass；新 launcher 一定要加跑 dataclass 检查。
- `train-dsh-minimal.sh` 强制要求 DSH gateway；非 DSH 的 harness 走官方 `scripts/code/train.sh`。
- TB2.1 是对方的评测基准，只用来验证链路，checkpoint 不保留。
- 冒烟时 GPU 平均利用率只有 50–66%，26–43% 的时间在空闲（在等 Modal 沙箱）。正式训练要提高并发轨迹数，或改用 separate_async。
- 单步跑 2 题 × 4 条，很容易整组得分相同，导致 grad 为 0。正式训练前要打开 DAPO 过滤，并把 batch 加大。
- Harbor 的参考解依赖 `/solution` 目录下的兄弟文件，oracle 必须把整个 `solution/` 上传。

## E. 第二轮决策与待办（2026-10-03）

用户已定：
- 两个 harness（mimocode、DSH）都是目标：训练混合保留两者，评测两者都要报。
- 第二轮上下文升到 96K：MAXLEN 98304、RESPONSE 94208、PPO_MAX_TOKEN_LEN_PER_GPU 98304、SGLang context_length 98304、DSH context_window 98304；单轮输出上限保持 32768。

待办：
- [ ] 96K 可行性冒烟：1 步，检查显存（64K 时峰值 48/96 GB）、SGLang KV 压力和单步耗时
- [ ] 评测口径统一到 96K（含 SFT 基线），保证第一轮和第二轮用同一把尺子
- [ ] 评测 k 值：TB2.1 主指标 mean@8 × 两个 harness，Code holdout100 mean@4 × 两个 harness（单 checkpoint 约 2.2K 条轨迹，4 卡约 12 小时）
- [ ] 评测 pod（等用户批准）
- [ ] batch1 难度 pilot（200 题 × 4 次）→ 决定全量预筛
- 依据：`docs/xdan-verl-fusion/data-scaling-analysis.md`
