# Ornith-1.5-9B 官方证据与横向比较输入

核查日期：2026-09-28。通过 gstack browse 读取以下官方页面。本文是证据整理，不代表复现成功。

## 来源

- O1：9B 模型卡 https://huggingface.co/ornith-ai/Ornith-1.5-9B
- O2：1.5 技术博客（Aug. 2026）https://ornith.ai/ornith_1_5.html
- O3：1.0 技术博客（Jun. 2026）https://ornith.ai/ornith_1_0.html
- O4：官方仓库 https://github.com/ornith-ai/Ornith-1
- O5：HF 组织及数据集入口 https://huggingface.co/ornith-ai
- O6：官方 GitHub 仓库列表 https://github.com/orgs/ornith-ai/repositories

## 结论与证据强弱

Ornith 是独立于 MiMo 的另一条路线，不能画成 MiMo SFT → Ornith。Ornith-1.5 的核心增量是将 1.0 的“生成 scaffold + 解题 rollout”扩展为“生成 task + scaffold + rollout”，三个阶段用各自奖励做 GRPO。发布了最终模型权重和方法说明，但已核查的官方入口未发现可复现的 1.x 训练程序与训练数据包。它适合成为推理候选模型和研究方法参考，尚不能称为公开完整训练配方。[O1/O2/O4/O5]

CPT、mid-training、post-training 的描述适用于 Ornith-1.0 家族背景；官方未披露 9B 每阶段独立的数据配方、步数、token 量和消融。不要将“post-training”擅自展开为一个已确认存在并有完整细节的 SFT 阶段，也不要把 Ornith-1.0 称为“SFT baseline”。[O1/O2]

9B 模型卡标记 qwen3_5 架构；1.0/1.5 家族介绍提到 Qwen3.5 与 Gemma4，9B 表中对照 Qwen3.5-9B。可以写“9B 属于 Qwen3.5 架构谱系”；不能写“9B 同时混合了 Qwen 和 Gemma 权重”，也没有证据称它使用 MiMo 教师蒸馏。1.5 延续 1.0，但未看到逐阶段 checkpoint 继承和更新的完整审计记录。[O1/O2/O3]

## 全流程：从 1.0 到 1.5

### 家族前置训练

官方称 1.0 在预训练 Qwen3.5 / Gemma4 上增加 continued pretraining、mid-training、post-training。阶段名称已披露，训练数据源、规模、token、混合比例、过滤方式及 9B-specific 配方未披露。[O1/O2]

### Ornith-1.0：两阶段自生成执行策略

1. 输入任务与此前用于该任务的 scaffold。
2. 模型提出改进的 scaffold。
3. 以任务与 scaffold 为条件生成解题轨迹。
4. 将 rollout 奖励传到两个阶段，使模型同时改进解题与组织解题的方法。[O3]

scaffold 是模型可修改的内部策略、记忆、错误处理与编排；不能理解为模型可以任意修改外部测试与信任边界。官方反作弊分三层：外部环境、工具面、测试隔离不可变；确定性监视器检测隐藏路径读取、改 verifier 等越界行为（违规轨迹记 0，且排除 advantage 计算）；冻结 LLM judge 充当 verifier 之上的否决器，非主要奖励。[O3]

异步训练：使用 pipeline-RL，并依据 token 陈旧程度 dt 做权重。dt ≤ K1 时 w=1；K1<dt≤K2 时 w=exp[-λ(dt-K1)]；dt>K2 时 w=0。token-level GRPO clipped objective 乘此权重。K1/K2/λ 的具体数值未给出。此为 1.0 方法披露，1.5 没有足够参数细节证明所有设置原样沿用。[O3]

### Ornith-1.5：三阶段自生成课程

输入：环境或代码库、任务类型的高级指令、此前解题历史。

1. **Task proposer**：提出超出已掌握能力、更难的新任务。
2. **Scaffold builder**：为任务构造或改进指令、工具组织、分解策略和编排。
3. **Solver**：依据任务与 scaffold 执行解题 rollout。
4. **Verifier / reward**：分别评价任务质量、harness 质量、解题质量。
5. **GRPO**：三个阶段用各自奖励联合改进；新策略再次生成新课程。[O2]

这是线上自生成训练经历机制，不等于没有初始数据，也不等于所有任务无需外部环境与验证器。

## 奖励设计

### 任务奖励

R_task = V(q,s) × D(q,s,{τ_i}) × N(q)。[O2]

- V∈[0,1]：有效且可验证。scaffold 能执行，可信正确解应通过，明显错误解应失败，评价符合任务规格。V=0 则任务奖励=0。
- p：N 次 rollout 的经验成功率。
- D=exp[-(p-p*)²/(2σ²)]，p*=0.2。偏好约 20% 当前成功率的前沿任务，既不是简单题，也不是完全无解题。σ 的具体值、每题采样 N 未披露。
- N(q)=1-max(sim(q,q_j))，q_j 来自历史任务缓冲区 B。相似度计算实现、embedding 模型、buffer 容量未披露。

### Harness 奖励

R_harness=C(q,h) × F(h,{τ_i}) × H(h)：任务对齐、奖励对真实质量的忠实性、抵抗奖励作弊。各因子的完整可运行评价实现未公开于所查入口。[O2]

### Rollout 奖励

R_rollout(τ_i)=h(q,τ_i)。可验证题可用二元 pass/fail；复杂环境可组合正确性、完成度、效率、约束满足。没有公开每个领域的权重配方。[O2]

## 数据调查：已知与未披露

| 层级 | 已知 | 未披露/未发现 |
|---|---|---|
| 原始基础模型 | Qwen3.5/Gemma4 家族背景，9B qwen3_5 架构 | 9B 精确初始 checkpoint 与所有中间版本 |
| CPT | 家族做过 CPT | 数据集名、许可证、来源、token 数、配比 |
| Mid-training | 家族做过 mid-training | 目标函数、样本格式、与 SFT 的边界、规模 |
| SFT | 未看到独立可审计阶段配方 | 教师名单、轨迹数、筛选方法、是否蒸馏 |
| RL 种子 | 输入环境/代码库与高级任务指令 | 仓库列表、环境数量、seed task 来源 |
| 动态 RL 数据 | 模型自生成 task、scaffold、rollout；历史缓冲区用于去重 | 总任务数、生成规模、去重实现、训练集清单 |
| 污染控制 | 评测时 SWE 去 git 历史且禁网；NL2Repo 部分资源屏蔽 | 完整训练集与基准去污染流程、重合审计 |
| 开放数据 | HF org 可见 CUDA-L1/L2 两个数据集 | 没有发现 Ornith 1.0/1.5 数据发布 |
| 训练实现 | 官方博客公式与机制 | Ornith-1 根目录仅 assets、.gitignore、LICENSE、README；无训练代码 |

必须避免把 CUDA-L1/L2 数据直接算作 Ornith 9B 训练数据：同组织发布不能证明被使用。基准名也不能当训练集名。[O2/O4/O5]

## 9B 同路线成绩（官方报分）

| Benchmark | Qwen3.5-9B | Ornith-1.0-9B | Ornith-1.5-9B | 1.5−1.0 |
|---|---:|---:|---:|---:|
| Terminal-Bench 2.1 / Terminus-2 | 21.3 | 43.1 | 46.2 | +3.1 |
| Terminal-Bench 2.1 / Claude Code | 18.9 | 40.6 | 47.0 | +6.4 |
| SWE-bench Verified | 53.2 | 69.4 | 70.6 | +1.2 |
| SWE-bench Pro | 31.3 | 42.9 | 47.5 | +4.6 |
| SWE-bench Multilingual | 39.7 | 52.0 | 54.4 | +2.4 |
| NL2Repo | 16.2 | 27.2 | 32.4 | +5.2 |
| SWE Atlas QnA | 9.2 | 17.9 | 20.6 | +2.7 |
| HLE / no tools | 14.7 | 16.8 | 20.2 | +3.4 |
| HLE / with tools | 24.5 | 26.4 | 30.5 | +4.1 |
| GPQA Diamond | 81.7 | 82.5 | 86.4 | +3.9 |
| MCP-Atlas | 46.8 | 49.4 | 54.2 | +4.8 |
| Toolathlon-Verified | 29.6 | 33.4 | 41.2 | +7.8 |
| WideSearch | 53.6 | 55.8 | 59.5 | +3.7 |
| BrowseComp | 41.5 | 44.8 | 56.4 | +11.6 |
| ClawEval | 53.2 | 63.1 | 66.5 | +3.4 |

来源 O1/O2。全部 1.5 成绩为 5 次独立运行平均；不是 best-of-5，也不是 pass@5。阶段总分差无法单独归因于 task generation，缺少固定其他条件的消融。

## 评测口径（横向比较必写）

- SWE Verified/Pro/Multilingual：OpenHands，temperature 1.0，top_p .95，256K；去掉 Git 历史、禁网。[O1]
- Terminal-Bench 2.1 / Terminus-2：Harbor/Terminus-2，parser=json，temperature 1.0，top_p 1.0，128K，4小时 timeout，32 CPU / 48GB RAM；调整 chat template 和 reasoning_content 适配。[O1]
- Terminal-Bench 2.1 / Claude Code：2.1.126，temperature 1.0，top_p 1.0，max_new_tokens 131072。[O1]
- Toolathlon：官方评测服务，token limit 128K。[O1]
- Ornith 70.6 与 MiMo RL 66.2 不能当同条件胜负。同基座在两个报告的 Verified 53.2 vs 60.0 已提示协议差异；不应做误差线缺失的显著性判断。

## 横向方法比较建议

| 维度 | MiMo（应由主报告源码审计确认） | Ornith（官方可核实） |
|---|---|---|
| 主要增量 | 蒸馏 SFT 起点上领域 GRPO | 自生成任务、scaffold、rollout 三阶段 GRPO |
| 数据机制 | 已发布任务集/环境记录 | 环境、指令、历史驱动动态生成，未发现训练集发布 |
| 控制变量 | 外部任务与评分配置 | proposer、scaffold、solver 共同优化；外边界应固定 |
| 成果形态 | 多领域实验，不能合并成单个万能 checkpoint | 发布 Ornith-1.5-9B 最终权重及该模型多项成绩 |
| 复现实用性 | 训练代码/任务资源可作为工程起点 | 方法可借鉴；缺完整数据和实现导致训练复现门槛未知 |

实用路径：先对 MiMo SFT 与 Ornith 最终模型做同 harness、同预算的业务基准；如需训练再利用 MiMo 开放管线建立静态任务闭环；稳定后只引入 Ornith 式任务生成，并保持独立 verifier。不应第一天同时重写任务生成、scaffold 与 evaluator，否则无法归因且奖励作弊风险放大。此为分析建议，非官方复现结论。
