# MiMo Agentic RL 源码、资产与复现审计

核查日：2026-09-28。方法：官方 GitHub API、Hugging Face API、官方技术报告 §7；仅源码审阅，未运行 GPU 训练。verl 固定提交 `a2ad9f6160b03ff2d47e59832bfb6b289f37c917`（mimo-oss）；数据版本 `639865fd3374018d6cb29b9fb82dd531406fcf5f`；9B 模型版本 `2367e865d009c13ac81713a2878291d33ab28177`。

## 1. 定位与开放边界

这是一套以现有 SFT checkpoint 为起点、覆盖五域环境和奖励的 Agentic RL 参考实现。不是从原始数据到蒸馏到五域统一成品模型的一键流水线。

- [verl README](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/README.md)：基于 verl 0.9.0.dev，推荐 MiMo-V2.6-Distill-Qwen-9B。
- [9B model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/blob/main/README.md)：明确发布的是 SFT checkpoint，Qwen3.5-9B 经 MiMo 生成数据监督微调。
- [9B 模型文件 API](https://huggingface.co/api/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)：公开、非 gated；单套四分片 safetensors、tokenizer、chat template；未见 RL 子目录。
- [官方模型列表 API](https://huggingface.co/api/models?author=XiaomiMiMo&search=MiMo-V2.6&limit=100)：核查时包含 9B Distill 与 Pro/Flash RL、Pro/Flash MOPD；未找到单独发布的 9B 各域 RL 成品权重。表述应为“未找到”，不据此断言未来不会发布。
- 全量 SFT 数据只披露了组成；不能将 RL-oss 任务集称为 77.4B-token SFT 数据。
- Pro/Flash 的 mixed RL / MOPD 结论不得移植到 9B 单域 GRPO 实验。

## 2. 公开任务数量与数据形态

来源：[数据卡](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/blob/main/README.md)、[viewer size API](https://datasets-server.huggingface.co/size?dataset=XiaomiMiMo%2FMiMo-V2.6-RL-oss)、[公开文件 API](https://huggingface.co/api/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss)。size API 返回 partial=false、pending/failed 为空。

| 配置 | 实际公开 train 行数 | 文件 | 论文近似任务数 |
|---|---:|---|---:|
| code | 2,698 | code.parquet | 3k |
| cyber | 1,000 | cyber.parquet | 1k |
| general | 989 | general/train.parquet | 1k |
| webdev | 2,093 | webdev.parquet | 2k |
| music | 1,000 | music.parquet | 约 1k（正文） |
| 合计 | 7,780 | 五个 train split | 约 8k |

论文表5使用去重任务 ID 的近似计数；viewer 是公开文件的行数，两种口径不要强行相等。此次 size API 合计与各配置求和一致。

文件 API 有 41,292 个文件，`general/envs/` 下 925 个不同目录；这是目录数，不等于 General 989 行的已验证可运行覆盖率。需逐行检查 env_task_dir 是否存在，不能假定一行一目录，也不能未验证就断言丢了64项。

General 公开清单中，非 env 文件仅 `general/train.parquet`。未见脚本注释提及的 `docker/build.sh`、`docker/retag_parquet.py` 或独立 `eval_300_open.retagged.parquet`。公开数据树与参考 launcher 的 open_source_env bundle 布局存在差异。

[image-mapping.jsonl](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss/blob/main/image-mapping.jsonl) 有 3,764 条映射，字段 `dataset_image` / `dockerhub_image`。例：`arvo-rl:v1-arvo-10055` → `docker.io/xiaomimimo/mimo-v2.6-rl-oss:arvo-v1-10055`；`general-agent-env-0:oss` → `docker.io/xiaomimimo/mimo-v2.6-rl-oss:general-agent-env-0`。这涉及 repository/tag 改名，不能只加 registry prefix。

## 3. 论文 §7：SFT 与 RL 应分开

来源：[技术报告](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)，印刷页33–36，表4–7。

表4：SFT 共77.4B total tokens /27.2B loss tokens。Code 23.2/7.3B、Cyber 11.0/4.8B、General 22.0/5.7B、Visual 21.2/9.4B。由同一个 SFT checkpoint 分别开展各域 GRPO；没有声明这里训练一个统一五域9B RL checkpoint。

表6（Base → SFT → 单域 RL）：

| Benchmark | Metric | Qwen3.5-9B | SFT | RL |
|---|---|---:|---:|---:|
| SWE-bench Verified | avg@3 | 60.0 | 61.1 | 66.2 |
| SWE-bench Pro | avg@3 | 32.0 | 44.6 | 47.6 |
| MiMo Code mini | avg@3 | 19.5 | 51.6 | 59.9 |
| MiMo Cyber mini | avg@3 | 5.7 | 31.3 | 47.0 |
| AutomationBench v1.0.6 | avg@1 | 5.0 | 30.3 | 33.1 |
| Terminal Bench 2.1 | avg@1 | 27.0 | 37.1 | 52.8 |
| Toolathlon-Verified | avg@1 | 25.9 | 35.2 | 38.0 |
| OfficeQA Pro | avg@1 | 9.0 | 19.5 | 24.8 |
| JobBench | avg@1 | 2.6 | 18.3 | 25.2 |
| MiMo General mini | avg@1 | 28.5 | 62.2 | 70.6 |
| MiMo Visual Coding mini | avg@1 | 61.7 | 64.0 | 72.4 |

音乐另在正文披露内部评分45.7→52.5。表6 Code 为 single-harness；表7是另一次 multi-harness 实验：4个训练 harness、3个 held-out harness，21个 dataset–harness pair 均提升。7个 harness 的无权平均 SFT→RL：Verified 62.3→65.7，Pro 44.4→46.5，Code mini 53.1→59.0。这些均值不可与表6单 harness值拼接。

## 4. 实际训练实现

五条 recipe 的 `adv_estimator=grpo`，SGLang rollout、Megatron actor。调用入口名 `main_ppo` 或 Cyber 的 `reward_manager=dapo` 不代表实际 estimator 是 PPO/DAPO。

| 领域 | wrapper 默认 GPU 拓扑 | rollout N | max tokens | trainer mode |
|---|---|---:|---:|---|
| Code | 8×8 | 16 | 262144 | colocate_async |
| Cyber | 8×8 | 16 | 262144 | colocate_async |
| General | 4×8 | 8 | 262144 | sync |
| Visual | 8×8 | 8 | 262144 | colocate_async |
| Music | 8×8 | 8 | 116384 | colocate_async |

这只是参考默认值，绝非最低卡数。未实测缩配、显存、耗时、预算，不估算最低硬件。General 的 rollout.mode=async 不改变 trainer mode=sync 这个事实。

源码：[Code](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/scripts/code/train.sh)、[Cyber](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/scripts/arvo/arvo.sh)、[General](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/scripts/general/general.sh)、[Visual](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/scripts/design/webdev.sh)、[Music](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/scripts/design/music.sh)。

### Code

路径：verl → uni-agent gateway / TransferQueue → mimoagent harness → K8s task pod → verifier reward → GRPO update。

- [四 harness spec](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/config/agent/code/mix-four-whitebox.yaml)：mini-mimocode、mini-bash、mini-claude-code、mini-codex；这是 harness 名，不是4个商业模型。mini-codex 配置的 model_name 为 policy。
- step-hash：每个group同一个harness，样本跨step轮换。N=16、batch=64、200 steps、LR=1e-6、prompt-mean、不按std归一化advantage、group filter开启、off-policy threshold=2。
- 基础 [train.yaml](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/code/config/train.yaml) 是4×8、CP=2、batch32；wrapper 覆盖成8×8、CP=1、batch64。env注释session=1024/timeout=7200，实际wrapper=2048/4800。以最终resolved_config为准。
- [run_train.sh](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/code/run_train.sh)检查集群、解析并校验Hydra、写manifest，支持PREFLIGHT_ONLY。
- [reward.py](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/code/reward.py)是fallback；正常reward来自runner session。
- [实际环境](https://github.com/XiaomiMiMo/mimoagent/blob/467f0a19016f0ac4d63b8d17a1f0da9ba07f232c/src/mimoagent/environments/datasets/opensource_code.py)：八字段合同为dataset_type、docker_image、cwd、instance_id、problem_statement、test_patch、test_command、verifier_timeout_sec。奖励阶段先reset隐藏测试路径，再apply test_patch，再执行test_command；exit0得1，否则0。镜像需截断git历史且不预放答案/隐藏测试。

### General

直接 [AgentLoop](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/general/agent_loop.py)驱动mimoagent，不经uni-agent。模型调用进入verl token接口；工具通过持有pod的Ray actor执行。

- [env示例](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/scripts/general/general.env.example)：每rollout一个双容器pod，需K8s、镜像仓库、envs目录、OpenAI兼容judge。示例MODEL_PATH=Qwen/Qwen3.5-9B，与README推荐的MiMo SFT不同；脚本没有固定模型默认值。
- batch64、N8、LR2e-6、5 epochs；reward默认二值化阈值1.0，infra用-999 sentinel，length penalty开启。
- [run_general.sh](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/general/run_general.sh)：要求train batch等于mini batch；校验agent_name和镜像前缀；sentinel与DROP_INFRA_FROM_GROUP=1不能叠加。
- Judge不可达可能出现全零reward，看起来像policy collapse，必须区分infra、judge、任务失败。

### 其他域

- [Visual env](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/scripts/design/webdev.env.example)：训练组内排名奖励，评估绝对评分，不可比较；需K8s、POD_PROXY、group grader、视觉judge；WEBDEV_DEBUG_DIR必须workers和driver共享，否则可能全零reward；渲染用HTTP避免file协议阻断模块。
- [Music env](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/scripts/design/music.env.example)：单轮ABC符号音乐，abc2midi与CPU scorer；启动worker probe防止缺二进制造成全零reward；CP=1的注释说明该模型线性attention context sharding未验证。无需mimoagent/uni-agent。
- [Cyber config](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/arvo/config/arvo.yaml)：GRPO、colocate_async、off-policy threshold8、300 steps；ARVO任务镜像与K8s。

## 5. 集成教程：可执行入口和待补条件

以下命令是源码对应的启动入口，未在本次环境执行训练；只有前提全部满足才可运行。不能把资产下载、配置检查通过写成论文分数复现成功。

### 5.1 固定源码和训练容器

```bash
git clone --branch mimo-oss https://github.com/XiaomiMiMo/verl.git
cd verl
git checkout a2ad9f6160b03ff2d47e59832bfb6b289f37c917
git submodule update --init third_party/mimoagent-osr third_party/uni_agent
# 在已匹配 CUDA / PyTorch / SGLang / Megatron 的训练容器中执行
pip install --no-deps -e .
```

最后一行只安装verl本体，不能补齐依赖。mimoagent README要求Python3.12；verl pyproject仅要求>=3.10，不能只满足后者。

[docker/README](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/docker/README.md)列出通用verl镜像sgl0512.latest / vllm024.latest；MiMo README链接的xiaomimimo/mimo-v2.6-rl-oss主要由任务image mapping引用，不应未经验证当训练基础镜像。

[requirements.txt](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/requirements.txt)约束transformers>=5.5.3,!=5.6.0,<5.11、TransferQueue==0.1.8。[通用安装脚本](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/scripts/install_vllm_sglang_mcore.sh)却固定sglang0.5.2、vllm0.24.0、FlashAttention2.8.3、Megatron core0.13.1、TE2.6，并安装/改变torch关联包、opencv、cuDNN，源码编译MAX_JOBS=32。Docker README的SGLang为0.5.12，说明这些入口不是一致的MiMo锁文件。不要在工作机现有环境盲跑该脚本；在隔离训练镜像中核对、固定digest并保存pip freeze。没有核实一套官方MiMo专用完整训练镜像digest。

### 5.2 数据准备与镜像适配

1. 下载固定版本数据和9B SFT模型到所有Ray节点相同可见路径；General还需下载完整envs树，不能只拿parquet。
2. 按image-mapping把每行实际环境镜像解析到可拉取地址；需检查行内extra_info / instance_json的结构，不能以字符串全局替换代替数据转换。
3. Code的docker_image合同是完整地址；可重写为dockerhub_image或镜像到自己的registry。General launcher使用DOCKER_REGISTRY加前缀，且拒绝别家完整域名；需保持“裸repository:tag + 本方prefix”或者修改配置适配已解析全名，避免双前缀。
4. 对General逐行检查env_task_dir在GA_TASK_ROOT下存在，保留原始数据与转换产物，记录映射/缺失清单。
5. 公开数据卡只注册train；自己制作无泄漏holdout并明确不是内部论文测试集。参考VAL路径不存在时必须覆盖，不能把train原样当val报告效果。

未找到公开的一键image-mapping转换器或文档所述General retag脚本；教程应将转换列为真实集成工作，不能编造官方命令。

### 5.3 集群先决条件

先准备已运行Ray集群；launcher默认RAY_INIT_ADDRESS=auto，不负责创建集群。每节点同路径可见模型、parquet、kubeconfig、源码/子模块；GPU可用、相关模块可import。K8s身份需创建/管理任务pod以及exec，节点能拉镜像；Visual需浏览器网络和共享截图目录。

[cluster_precheck.py](https://github.com/XiaomiMiMo/verl/blob/mimo-oss/recipes/code/cluster_precheck.py)在各GPU节点probe CUDA、路径及verl、megatron.core、megatron.bridge、fla、sglang、mimoagent；处理镜像/宿主CUDA driver不一致。文内torch2.11+cu130/driver570是作者实际环境背景，不是普遍硬件要求。

### 5.4 Code配置与preflight

```bash
cp scripts/code/env.example scripts/code/local.env
# 编辑local.env：MODEL_PATH、TRAIN_DATA、VAL_DATA、KUBECONFIG
# 必须使用转换后且已验收的数据；根据实际资源设置TRAIN_NNODES等
set -a
. scripts/code/local.env
set +a
PREFLIGHT_ONLY=1 bash scripts/code/train.sh
# 仅在worker probe、resolved config、manifest均正确后启动
bash scripts/code/train.sh
```

MODEL_PATH建议使用固定revision下载后的本地共享路径：precheck用os.path.exists验证它。默认RUN_DIR=outputs/four-whitebox/<UTC-run-id>；保存resolved_config.yaml、checkpoint、rollouts、validation、trajectories及manifest。可以显式固定RUN_DIR方便检查。PREFLIGHT_ONLY仍会连接Ray、使用GPU probe并写输出，不是纯静态检查。

### 5.5 General配置、preflight与eval

```bash
cp scripts/general/general.env.example scripts/general/general.env
# 编辑GA_TASK_ROOT、MODEL_PATH、KUBECONFIG、DOCKER_REGISTRY
# 编辑GA_JUDGE_URL、GA_JUDGE_KEY、GA_JUDGE_MODEL
# 额外显式export TRAIN_DATA和VAL_DATA到真实准备好的文件
source scripts/general/general.env
PREFLIGHT_ONLY=1 bash scripts/general/general.sh
bash scripts/general/general.sh
# 使用明确holdout，显式覆盖VAL_DATA与checkpoint
GENERAL_MODE=eval bash scripts/general/general.sh
```

General默认找GA_TASK_ROOT/parquet/train_1000_open.retagged.parquet和eval_300_open.retagged.parquet，公开资产并非此布局，必须覆盖TRAIN_DATA/VAL_DATA。MODE=eval切val_only、val_before_train，使用VAL_KWARGS_N（默认4）采样。确保MODEL_PATH指向可加载模型格式；不能假定Megatron训练checkpoint目录直接等价HuggingFace格式。本轮未验证导出/合并流程。General preflight主要校验数据与解析Hydra，不执行Code同等强度的逐节点GPU probe，也不证明judge/任务pod成功。

### 5.6 复现验收

先用一条任务验证pod创建、工具、模型响应、最终评分与清理；再小批数据确保有有效reward、没有全零/infra哨兵污染；完成一次权重更新、保存与恢复；固定holdout、judge、harness、采样和窗口测量前后变化。最后扩规模并记录吞吐、成功率、infra失败率、版本与成本。每阶段是集成验收建议，不是已实测结果。

当前公开证据不足以承诺：一键复现所有论文内部benchmark、拿到所有9B RL最终权重、重建完整77.4B-token SFT阶段、用指定最低GPU数量复现。可复现目标应先限定为“从公开SFT起点，跑通一个领域可审计的RL闭环”。
