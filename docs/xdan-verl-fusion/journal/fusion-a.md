# 融合线 A 组训练日志（group-a-r1）

分级规则见 [README.md](README.md)。时间一律为 UTC。

## 运行概况

| 尝试 | W&B | 结果 | 原因 |
|---|---|---|---|
| attempt1 | — | 弃用 | 误用 `train-dsh-minimal.sh` 的 2 卡默认值，吞吐只有 23 tok/s |
| attempt2 | ga101 | 弃用 | W&B id 被复用，`resume=never` 导致启动失败 |
| attempt3 | ga102 | 跑到 step 7 后停止 | DSH 单轮输出上限 4096 的 bug（见下文 10-02 分析） |
| **ga103** | ga103 | 进行中，10-02 18:32 开跑 | source-5d234637；100 步，每 5 步存盘，64K，DSH+mimocode 混合 |

---

## 分析记录（L3，回填）

### 2026-10-02 ~17:00 分析：DSH 偏弱的根因是单轮输出上限 4096

- 现象：attempt3 第 1–7 步，DSH 的原始成功率约 0.36，mimocode 约 0.60。
- 根因：DSH 的 mixed profile 沿用了 minimal 脚本的 `max_tokens: 4096`，mimocode 则是 r2 的 32768。`train.sh` 只会改写官方 four-whitebox 组合的上限，管不到自写的 mixed profile。
- 证据：322 个 DSH 会话里，155 个以 `turn/end reason=max-tokens` 结束（48%），平均分 0.13；正常完成的 DSH 会话平均 0.63，和 mimocode 的 0.60 持平。
- 修复：`626c5647` 和 `5d234637`。单轮上限改为 32768，run_timeout 改为 4800，命令超时改为 300；context_window 保持 65536（等于 SGLang/MAXLEN 的上限）。两个 profile 的环境块和模型块现在逐行一致。
- 验证 run（只用 DSH，4 题 × 8 次，1 步）：reward 0.56，撞单轮上限 0 次，10/27 个会话上下文用满，exit 0。
- 用户决定：停掉 attempt3，重新开跑（ga103），并把存盘改成每 5 步一次，使 25/50/100 的评测点都有 checkpoint。

### 2026-10-03 00:50 分析：修复生效，DSH 与 mimocode 的差距从约 24pt 缩小到约 7pt

| 前 8 步平均 | attempt3（修复前） | ga103（修复后） |
|---|---|---|
| 每步耗时 | ~44 min | 44.5 min |
| 被接收组 reward | 0.49 | 0.55 |
| DSH 原始成功率 | 0.36 | 0.47 |
| mimocode 原始成功率 | 0.60 | 0.54 |
| DSH 撞单轮上限 | 48% | 0 |

- 看门狗误报：DSH 用 `max-tokens` 同时表示「单轮输出撞上限」和「64K 上下文用满」。改为按最后一轮的 usage 区分（输出 ≥30000 才算撞上限）后，撞上限次数为 0。

### 2026-10-03 ~01:10 分析：上下文用满和 Harbor 太容易

- 上下文用满（64K）：mimocode 24.5%（111/453），这些会话平均 0.48；DSH 36%（142/391），平均 0.33（Code）或 0.40（Harbor）。用满后仍会判分，因为沙箱里已有的修改照样算数，所以并不是 0 分。DSH 首轮输入只有约 1.4K token（mimocode 约 7K），推断它是在会话过程中积累得更快（历史推理内容或工具输出没有截断），**尚未验证**。
- Harbor stage1 对这个模型太容易：mimocode 在 Harbor 上平均 0.94，整组全对，被 DAPO 过滤掉。前 8 步只接收了 3 个 Harbor 组。之前「Harbor 太难」的猜测是错的。

### 2026-10-03 05:05 分析：吞吐和时间预算（供评测排期使用）

- 每步耗时拆分（第 9–14 步平均）：gen 1514s（59%），old_log_prob 221s（9%），update_actor 760s（30%），每 5 步存盘一次 59s。
- rollout 吞吐：约 198 会话/小时（4 卡，2 个 TP2 副本，并发 48），约 50 会话/GPU·小时。
- 每条轨迹模型实际生成的 token：mimocode 平均 20.3K，DSH 平均 23.8K。response_length 约 45K，但它包含工具输出，不能拿来估算生成量。之前「双卡 13–19 小时」的估算就是误用了它，已作废。

### 2026-10-03 ~06:30 分析：最终目标升级为超过 Qwen3.5-9B 和 Ornith-1.5-9B

- SFT 起点 MiMo-V2.6-Distill-Qwen-9B 是在 Qwen3.5-9B 上做的 SFT（见 model card）。按 MiMo 报告的口径，TB2.1 从 27.0 到 37.1，SWE Verified 从 60.0 到 61.1，SWE Pro 从 32.0 到 44.6。
- Ornith 的公开口径：TB2.1 46.2（Terminus-2，128K），SWE Verified 70.6（OpenHands，256K）。口径不同，只能看量级：差距约 9–10pt，是原定 +3pt 目标的 3 倍。
- 推断：第一轮的配置跨不过这个差距。第一轮的定位改为「验证方法、测出 RL 的提升斜率」。第二轮要考虑上下文 ≥128K、第三方 harness、对准榜单的数据、多领域（MOPD）和算力。等公开口径的 SFT 分数出来后再定。

### 2026-10-03 07:44 分析：与 baseline 线横向对比，以及 General 分数事故

- 两条线互补：baseline 横向铺开五个域，回答「能不能跑通」；融合线纵向做深，回答「多 harness 长训练能不能在独立榜单上涨分」。最终的多维度目标需要两者结合（MOPD）。
- 事故（baseline 线）：General 4h 评测的分数是几百的负数。根因是 LLM judge 在 128 路并发下崩溃，失败样本记为 −999，又被平均进了 val-core。baseline 已在约 16:00 起重测。看门狗对超出 [0,1] 的均值统一标记为无效。教训：rc=0 不代表结果有效。

### 2026-10-03 10:03 分析：前 21 步训练健康，但还看不出效果

| 指标 | 第 1–7 步 | 第 8–14 步 | 第 15–21 步 |
|---|---|---|---|
| 原始成功率 | 0.53 | 0.50 | 0.52 |
| 被接收组 reward | 0.57 | 0.52 | 0.50 |
| 熵 | 0.41 | 0.41 | 0.43 |
| 平均响应长度 | 41K | 48K | 43K |
| 全错组占比 | 0.14 | 0.07 | 0.12 |

- 结论：原始成功率和熵都平稳，没有塌缩，长度也没有失控。被接收组 reward 的下降来自每一步抽到的题不同，不代表训练变差。训练指标也还没有显示出提升，要靠 step 25 和 step 50 的独立评测配对来判断。
- 其他：会话失败 1/248；DSH 撞单轮上限 0 次；checkpoint 5/10/15/20 已保存；SFT × TB2.1 × mimocode 评测已完成 709/712。

---

## 用户决策记录

| 时间 | 决策 |
|---|---|
| 10-02 | 训练目标：TB2.1 或 holdout100 中至少一项比 SFT 高 ≥ +3pt，另一项不退步；评测点 25/50/100；step 50 无增益则停 |
| 10-02 | 停掉 attempt3，修复 DSH 后重新开跑（ga103） |
| 10-03 | 两个 harness 都要；第二轮升到 96K；评测统一用 96K |
| 10-03 | 4 卡评测 pod 由用户开机，fusion-eval 会话负责评测；TB2.1 用 mean@8；增加 64K 对照组 |
| 10-03 | 预注册判定：TB2.1 strict mean@8，分 harness，配对差 ≥ +3pt 且 95% CI 下界 > 0（CI 条件来自训练会话的建议，用户可否决） |
| 10-03 | 最终目标：在多个榜单上同时超过 Qwen3.5-9B 和 Ornith-1.5-9B，并能横向对比 |

---

## 运行日志（L2，看门狗自动追加）

### 2026-10-03 10:40 分析：SFT 的 TB2.1 基线（mimocode）是 27.1%；上下文撑满主要是「卡住的长会话」，加长上下文救不回来

- 来源：fusion-eval 的 spec fusion-a-v1，96K，mean@8，共 712 个会话。strict 27.1%，gradable 27.2%；失败或超时 1.1%（计 0 分）。这个分数不能和 MiMo model card 上的 37.1 比，因为口径不同（harness、上下文、avg@1）。
- 结束原因：正常完成 74%，平均 0.342；上下文用满（LimitsExceeded）25%，平均只有 0.073。64K 训练时这个比例是 24.5%，**上下文加到 96K 后比例没有下降**。
- 在训练 pod 上只读拆解（中位数）：

| 结束方式 | 会话数 | 轮数 | 总 token | 模型生成 | 工具输出 | 模型生成占比 |
|---|---|---|---|---|---|---|
| 正常完成 | 527 | 33 | 59K | 33K | 13K | 0.70 |
| 上下文用满 | 177 | 76 | 98K | 54K | 34K | 0.61 |

- 结论：
  1. 用满上下文的会话是**轮数翻倍以上、迟迟不收敛的长会话**，不是几次超大的工具输出造成的。即使给到 96K，它们也只拿到 0.07 分，所以继续加长上下文的边际收益很低（推断，128K 下待验证）。
  2. 上下文的大头是**模型自己的推理**（约 60–70%）。所有轮次的推理都保留在历史里，正常完成的会话约 33 轮就用掉了约 60K。
- 对第二轮的含义（待用户决策）：「升到 96K」的收益可能小于预期。更有效的方向可能是：(a) 管理上下文，比如不保留历史轮次的推理，或者开启 harness 自带的压缩，但这需要训练端支持多段轨迹；(b) 让 RL 惩罚不收敛的长会话、奖励及时收尾，64K 训练本身就带有这种压力。动手之前，先看 Terminus-2 口径下用满上下文的比例，再看卡住的会话是否在重复同样的命令。

### 2026-10-03 11:10 分析：在 Terminus-2（Ornith 口径）下，SFT 主要失分于输出格式，而非能力

- 来源：fusion-eval 的集成窗口。SFT 用 SGLang 起服务，128K，Harbor + Terminus-2（parser=json），跑 TB2.1 的 3 题。结果 3/3 都在 AgentTimeoutError（900s）时结束，得 0 分。
- 根因：SFT 输出的是自己训练时用的原生工具调用格式 `<tool_call><function=commands>…`，不是 Terminus 要求的 JSON（analysis/plan/commands）。以 prove-plus-comm 为例：91 步里有 70 步（77%）是 JSON 解析失败；被提示修正格式后只改一两轮，又回到原格式，最后空转，输入 token 累积到 150 万到 1600 万。解题思路本身是对的。
- 含义：
  1. 按 Ornith 口径，SFT 的分数会被格式失败大幅拉低。Ornith 的 46.2 里包含了针对 harness 协议的适配训练。所以**「遵循第三方 harness 协议」本身就是一项需要训练的能力**。
  2. 我们的任务和 harness 是正交设计，加入一个新 harness 的成本低。第二轮把 Terminus 风格的 harness（JSON 动作协议，不传 tools 参数）加进训练混合，是可行而且对准目标的做法。
  3. 横向对比时，要把「格式失分」和「能力失分」分开：用 Terminus 的 xml parser，或者用原生工具调用的 harness（如 Claude Code、OpenHands 的 function calling）测一次 SFT。Ornith 用 Claude Code 是 47.0，这个口径可以直接对比。
- 相关信息：mimocode 口径下 SFT 是 27.1%；MiMo 报告的 37.1 是 avg@1，用的 harness 没有公开（列在 General 域下），三个数的口径各不相同。

### 2026-10-03 11:20 用户决策：在原生工具调用口径下测一次 SFT（选项 a）

- 目的：把 Terminus-2 下的「格式失分」和「能力失分」分开。
- 方案（已转给 fusion-eval）：首选 Harbor 的 claude-code agent（Ornith 在此口径下 TB2.1 为 47.0），需要在 SGLang 前加一层 Anthropic→OpenAI 的转换代理；接不通就退回 OpenHands 的 function-calling。先抽 15–20 题、每题 1–2 次做诊断，排在 SFT×TB2.1×DSH 之后、step 25 之前，不影响 step 50 的关键路径。
- 256K 定向上下文测试（选项 b）这次没有选。

### 2026-10-03 12:35Z 推送

**到达节点 step 25**：评测点 1：对比 SFT 基线（HF 权重已由 milestone_keeper 永久留存）

**进度** `▓▓▓▓▓░░░░░░░░░░░░░░░` 25/100（25%）
- 下一节点：step 50：决策点：对比 SFT，无增益则停
- SFT 基线（TB2.1，两个 harness）：⏳ 未完成（step 50 判定需要它）
- 已完成评测（strict）：sft-mimocode-tb21 0.271
- 已保存 checkpoint：[5, 10, 15, 20, 25]

### 2026-10-03 12:35Z 推送

**进度** `▓▓▓▓▓░░░░░░░░░░░░░░░` 25/100（25%）
- 下一节点：step 50：决策点：对比 SFT，无增益则停
- SFT 基线（TB2.1，两个 harness）：⏳ 未完成（step 50 判定需要它）
- 已完成评测（strict）：sft-mimocode-tb21 0.271
- 已保存 checkpoint：[5, 10, 15, 20, 25]

**近 5 步**
- 步时 43.1 分钟，预计剩余 54 小时
- reward 0.506，grad 0.223
- harness：mimocode 0.502 / DSH 0.504
- 被接收的组：Code 33 / Harbor 7
- DSH 结束原因（近 2 步）：{'completed': 23, 'context-full': 5}
- DSH 失败累计 3，ungradable 0，沙箱 48

### 2026-10-03 ~12:00 分析：「写不完」的收益要按题控制难度来估：对 DSH 很大，对 mimocode 很小

- 更正之前的说法。「DSH 正常完成的会话 0.50 > mimocode 0.34，所以减少写不完比提升能力更能拉分」有选择偏差：完成的会话本来就偏向简单题。
- 估算方法：按题计算。同一题内，把没写完的会话按该题已完成会话的通过率补上；一次都没写完的题记 0。得到的是收益上限。

| harness | 实际得分 | 没写完占比 | 全部写完的上限 | 收益上限 |
|---|---|---|---|---|
| mimocode（712 个会话，已完成） | 27.1 | 26% | 31.1 | +4.0pt |
| DSH（441/712，进行中） | 24.3 | 60% | 37.2 | +13.0pt（21 题从未写完，记 0） |

- 结论：mimocode 的主要瓶颈是解题能力；DSH 的主要瓶颈是「开局想太多」（16%）和「上下文用满」（37%）。
- 处理原则：不直接惩罚输出长度，以免模型学会提早放弃。保持只按结果给分；在 harness 层设单轮思考预算；优先做上下文管理（丢弃历史推理或压缩）。验证时看 end_shares 中的 turn_cap 和 context_full 是否下降，同时已完成会话的得分不下降。

### 2026-10-03 ~12:40 分析：两个 harness 都自带上下文管理，但在我们的配置里都没生效

- mimocode：
  1. 由模型调用的 `compact` 工具，没有出现在官方和我们的工具列表里；
  2. 上下文用量提示的分母 `compaction_context_window` 默认是 1M，在 64K/96K 下模型只看到 6–10%；
  3. `wrap_up_hint` 默认关闭；
  4. `max_observation_length` 为 0，工具输出不截断。
- DSH：`sdk-minimal` profile 按设计排除了 compaction（runtime 文档原文）。可以通过 profile patch 插入 compaction 组（compaction-basic、command-compact、tool-result-pruner），但 DshSdkAgent 把 `DSH_UA_PATCHES` 固定为 `[]`，需要改代码。
- 用户决策（「不动训练」）：只在评测侧用 SFT 做 A/B，单独登记 spec（harness-ab-v0），不改 fusion-a-v1，也不改训练。mimocode 的 A 组为加 compact 工具并使用真实分母，B 组为 A 加收尾提示和工具输出截断，已交给 fusion-eval。DSH 的 patch 支持等代码改好后再加入。
- 训练端的前提：上下文压缩会把一个会话切成多段轨迹（num_trajectories > 1）。第二轮要采用压缩，就得先验证训练端能正确处理多段轨迹。

### 2026-10-03 15:35 分析：SFT 的 TB2.1 基线齐了（mimocode 27.1%，DSH 21.1%）；作废规则需要用户确认

- SFT × TB2.1 × DSH（96K，mean@8，712/712）：strict 21.1%，gradable 21.9%。
  - end_shares：turn_cap 12.8%，context_full 44.5%，route_401 1.1%，dsh_run_timeout 6.5%。
  - 各结束原因的平均分：completed 242 个 0.55，context-full 317 个 0.054，其余为 0。
- 对照 mimocode：27.1%，完成率 74%，完成会话平均 0.34。DSH 只要跑完，得分更高（0.55），但约 2/3 的会话没跑完，所以 DSH 的提升空间几乎都在「收敛」上（与 12:00 的估算一致）。
- 训练进度：step 29。step 25 的 HF 权重已于 12:33 留存（18G）。
- 待决：评测 spec 写的是「n_failed > 5% 则 INVALID」。DSH 的失败共 8.7%，但其中大部分是超时、撞上限这类模型行为，本来就按 strict 计 0 分。建议只统计基础设施类失败（framework_error、沙箱或 Modal 错误、route_401，合计约 2.2%）。按这个口径，SFT 的 DSH 基线有效。需要用户确认。
