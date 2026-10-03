# Round-2 训练数据：Harbor batch1 按 TB2.1 类目分层（2026-10-04）

> 目标：Round-2 = MiMo Code 2,598 + Harbor batch1 子集（按 TB2.1 16 类目分布分层），替换 Harbor stage1 500 行。
> 本文所有数字可由 `scripts/data/relabel_tb21_taxonomy.py` + `scripts/data/select_tb21_stratified.py` 复现。

## 1. 产物

| 产物 | 位置 |
|---|---|
| 重标注脚本 | `scripts/data/relabel_tb21_taxonomy.py`（LLM 分类，JSONL 缓存可断点续跑） |
| 分层选择 + 合并脚本 | `scripts/data/select_tb21_stratified.py`（`select` 本地；`materialize` 在 pod 调 `combine_group_a.py`） |
| 1,851 条标签 | `docs/xdan-verl-fusion/data-r2/batch1-tb21-labels.jsonl`（sha256 `2356a1bd…`） |
| 选择结果（两变体 id + 统计） | `docs/xdan-verl-fusion/data-r2/batch1-tb21-selection.json` |
| 合并 manifest（本地副本） | `docs/xdan-verl-fusion/data-r2/train-r2-{a,b}.manifest.json` |
| 训练 parquet（pod） | `/workspace/xdan-verl-fusion/data/group-a-r2/train-r2-a.parquet`（2,658 行，sha256 `507607cf…`） |
| | `/workspace/xdan-verl-fusion/data/group-a-r2/train-r2-b.parquet`（3,098 行，sha256 `4a085b59…`） |
| batch1 子集 / 输入副本 | `group-a-r2/batch1-tb21-{a,b}.parquet`，`group-a-r2/inputs/`（脚本、标签、selection） |

合并沿用 `combine_group_a.py`（与 pod `source-9a68147f` 中的版本 sha256 一致）：deny-list
（`eval-denylist.txt`，1,682 行）复检、instance id 去重检查、seed 20261002 打乱。回读验证：两份 parquet
id 全唯一、deny-list 命中 0、schema 与 `group-a/train-r1.parquet` 完全一致、selection id 全部在内。

## 2. 方法

1. **重标注**：LLM 看 16 个类目名 + 我写的一行定义（未放任何 TB2.1 任务文本）+ 原生 category/tags
   + instruction 前 4,000 字符，输出 `{category, secondary_category, confidence, rationale, difficulty_hint}`。
   规则要点：语言不决定类目；领域明确时取最具体类目；GitHub issue 若是新行为/增强 → software-engineering，
   若是报告缺陷 → debugging。temperature 0，并发 8，失败重试（指数退避），`difficulty_hint` 与原生难度分开存。
2. **分层**：每类内部排序 = 优先级档（原生与 LLM 难度都 medium/hard → 0 档；仅一个 → 1 档；都 easy → 2 档），
   同档内 seed 20261004 洗牌。SWE 每 repo 上限 30，先截断再算可用量（实测不起作用：最大 repo
   cfn-lint/moto 恰好各 30，截断 0 条）。
   - (a) **strict-proportional**：无覆盖类目（games、personal-assistant）丢弃，其余 TB2.1 份额重归一；
     最大余数法分配，扫描所有 N 取「每类配额 ≤ 可用量」的最大 N。
   - (b) **cap-and-fill**（默认 total=500，即与被替换的 stage1 500 行等量）：按 16 类原始份额分配 →
     每类截到可用量 → 缺口按 TB2.1 份额在仍有余量的类目间迭代灌水。

## 3. 类目表（TB2.1 份额 vs batch1 可用量 vs 选中量）

| 类目 | TB2.1（份额） | batch1 可用 | (a) 选中（占比） | 500 的比例配额 | (b) 选中（占比） |
|---|---|---|---|---|---|
| software-engineering | 26 (29.2%) | 905 | 18 (30.0%) | 146 | 216 (43.2%) |
| system-administration | 9 (10.1%) | 170 | 6 (10.0%) | 50 | 74 (14.8%) |
| scientific-computing | 8 (9.0%) | 5 | 5 (8.3%) | 45 | 5 (1.0%) |
| security | 8 (9.0%) | 52 | 5 (8.3%) | 45 | 52 (10.4%) |
| data-science | 8 (9.0%) | 8 | 6 (10.0%) | 45 | 8 (1.6%) |
| debugging | 5 (5.6%) | 535 | 3 (5.0%) | 28 | 42 (8.4%) |
| file-operations | 5 (5.6%) | 24 | 3 (5.0%) | 28 | 24 (4.8%) |
| model-training | 4 (4.5%) | 3 | 3 (5.0%) | 22 | 3 (0.6%) |
| mathematics | 4 (4.5%) | 5 | 3 (5.0%) | 22 | 5 (1.0%) |
| data-processing | 4 (4.5%) | 73 | 3 (5.0%) | 22 | 32 (6.4%) |
| machine-learning | 3 (3.4%) | 21 | 2 (3.3%) | 17 | 21 (4.2%) |
| games | 1 (1.1%) | 0 | 0 | 6 | 0 |
| personal-assistant | 1 (1.1%) | 0 | 0 | 6 | 0 |
| optimization | 1 (1.1%) | 10 | 1 (1.7%) | 6 | 8 (1.6%) |
| data-querying | 1 (1.1%) | 38 | 1 (1.7%) | 6 | 8 (1.6%) |
| video-processing | 1 (1.1%) | 2 | 1 (1.7%) | 6 | 2 (0.4%) |
| **合计** | 89 | 1,851 | **60** | 500 | **500** |

与 TB2.1 分布的 total-variation 距离：(a) 0.049，(b) 0.266。
(a) 的 N=60 由 model-training（可用 3、份额 4/87）卡住；scientific-computing、data-science 也几乎用满。

(b) 的 total 敏感性（`selection.json` 的 `cap_and_fill_sweep`）：

| total | 60 | 300 | 500 | 800 | 1000 | 1500 |
|---|---|---|---|---|---|---|
| TV 距离 | 0.055 | 0.227 | 0.266 | 0.333 | 0.364 | 0.408 |

## 4. 家族与难度构成

| | (a) N=60 | (b) N=500 | TB2.1 |
|---|---|---|---|
| swe-rebench / terminal-lego | 17 / 43 | 162 / 338 | — |
| 原生难度 easy/medium/hard | 12 / 45 / 3 | 97 / 385 / 18 | 4 / 55 / 30 |
| LLM difficulty_hint easy/medium/hard | 10 / 50 / 0 | 70 / 418 / 12 | — |
| 优先级档 0 / 1 / 2 | 48 / 2 / 10 | 399 / 35 / 66 | — |
| 单 repo 最大任务数 | 2 | 9 | 上限 30 |

全量 batch1：原生 hard 仅 36（1.9%），LLM 判 hard 仅 55（3.0%）；TB2.1 hard 占 34%。**难度缺口比类目缺口更大**，
分层只能把 easy 挤到 ~20%，无法补出 hard。2 档（双 easy）主要来自 file-operations（24 条中 23 条 LLM 判 easy）
与 data-processing（73 条中 63 条 easy）。

## 5. 缺口报告（Gap）

| 类型 | 类目 | batch1 可用（主类） | 次类目命中 | 500 规模下缺口 |
|---|---|---|---|---|
| 零覆盖 | games | 0 | 0 | 6 |
| 零覆盖 | personal-assistant | 0 | 0 | 6 |
| 严重不足 | scientific-computing | 5 | 33 | 40 |
| 严重不足 | data-science | 8 | 26 | 37 |
| 严重不足 | model-training | 3 | 1 | 19 |
| 严重不足 | mathematics | 5 | 11 | 17 |
| 不足 | video-processing | 2 | 2 | 4 |
| 轻微不足（且偏 easy） | file-operations | 24 | 95 | 4 |
| 充足 | software-engineering / debugging / system-administration / security / data-processing / machine-learning / data-querying / optimization | — | — | 0 |

需要外部来源的类目（按优先级）：scientific-computing、data-science、model-training、mathematics（四类合计占 TB2.1 27%，
batch1 只有 21 条）；其次 games、personal-assistant、video-processing（各 1/89，体量小但零/近零覆盖）。
候选：terminal-lego-15k 其余未入 batch1 的任务按本脚本同样打标后补洞；或 batch2 定向挖掘
（scientific/numerical、pandas 分析、PyTorch 训练、数学/符号计算、ffmpeg、游戏/谜题）。次类目命中（如
scientific-computing 33 条）可作为低成本备选，但这些任务主技能并非该类目，不建议直接算作配额。
另外 hard 难度普遍缺失，需要来源侧解决（swe-rebench 的 major_bug、多文件任务，或更难的 terminal 任务）。

## 6. 标签质量

- **SWE 合理性**：swe-rebench 792 条 → debugging 481 / software-engineering 305 / optimization 4 / security 2，
  99.2% 落在 SE 或 debugging，符合预期。
- **原生 → 新类目混淆**（主类）：
  - software-engineering(792) → debugging 481, SE 305, optimization 4, security 2
  - general(559) → SE 331, sysadmin 66, security 47, data-processing 32, debugging 25, data-querying 22,
    file-ops 15, ML 9, math 5, 其余 ≤3
  - programming(367) → SE 231, data-processing 38, sysadmin 36, debugging 27, ML 12, data-science 7, 其余 ≤4
  - system-administration(33) → sysadmin 21, SE 9；database(24) → data-querying 14, SE 9；
    version-control(24) → sysadmin 15, SE 8；shell-scripting(20) → SE 11, file-ops 4, data-processing 3；
    containerization(13)/networking(12)/web-server(7) → 几乎全部 sysadmin
- **人工抽检 30 条**（seed 20261004 随机，逐条读 instruction）：主类目同意 **27/30 = 90%**；
  若接受「我的判断 = LLM 主类或次类」则 **30/30**。3 条分歧均为相邻类目：
  task_06029（C# CSV→TSV，LLM=SE，我判 data-processing）、task_10420（修复堆上执行段错误，LLM=SE，我判 debugging）、
  task_04964（matplotlib 标注图，LLM=SE，我判 data-science）。偏差方向一致：**边界任务倾向判 SE**，
  因此小类目的可用量是略被低估而非高估。另 task_11550（git 切分支 → sysadmin，conf 0.6）可接受但属边界。
- **置信度**：中位数 0.90，P10 0.72，<0.7 的 108 条（5.8%）。
- 质量可接受，未改 prompt 重跑（PROMPT_VERSION=v2 为唯一一次全量运行）。

## 7. LLM 调用与成本

- 模型：`deepseek-v4-flash`（1,827 条）+ 回退 `deepseek-v4-pro`（24 条）。**偏离原计划**：当前 LiteLLM key 无权访问
  `deepseek-chat` / `glm-5-turbo`（403，允许列表只含 gpt-5.6-*、deepseek-v4-*），故改用同系列可用模型。
- 调用：成功 1,851 次，总尝试 2,332 次（481 次重试，日志未逐条记录原因，推测为解析失败/瞬时错误）。
- Token（仅成功调用的 usage）：prompt 2,723,092，completion 765,607（含 reasoning）。墙钟约 30 分钟。
  失败尝试的 token 未计入；美元成本取决于 proxy 计价，未核实。

## 8. Code : batch1 比例

| 变体 | MiMo Code | batch1 | 总行数 | Code : batch1 |
|---|---|---|---|---|
| (a) strict-proportional | 2,598 | 60 | 2,658 | 43.3 : 1（batch1 占 2.3%） |
| (b) cap-and-fill(500) | 2,598 | 500 | 3,098 | 5.2 : 1（batch1 占 16.1%，与 r1 相同） |

## 9. 建议：选 (b)

- (a) 分布最贴近 TB2.1（TV 0.049），但 N 只有 60，Harbor 信号被 Code 稀释到 2.3%，且每个小类目只有 1–6 条，
  GRPO 训练中这些任务的统计意义很弱；实质上等于「去掉 Harbor」。
- (b) 与 r1 等量（500），只改变「替换哪 500 条」这一个变量，便于与 r1 对比；覆盖 14/16 类目，
  80% 为 0 档（双 medium/hard）。代价是 SE 偏高（43% vs 29%）、四个稀缺类目远低于目标——这是数据源限制，
  任何 total 下都无法靠重采样解决。
- 不建议把 (b) total 继续加大：800 以上 TV 迅速恶化，且新增的几乎全是 SE/debugging/easy。
- 若想在 (b) 内再贴近 TB2.1，可考虑对稀缺类目（model-training、scientific-computing、data-science、mathematics）
  做 2–3× 重复采样；本次未做，留给训练侧决定。

## 10. 未核实 / 注意事项

- 未使用 `deepseek-chat`（key 无权限），标签来自 `deepseek-v4-flash`。
- 人工抽检仅 30 条，95% 置信区间约 ±11 个百分点；小类目（scientific-computing 等）的精度未单独核验。
- 未在 pod 上用训练 dataloader 实际加载 r2 parquet（只做了 schema 与 r1 一致的校验）。
- TB2.1 类目统计取自 pod 上的 task.toml，于本次重新核对一致（16 类、89 题、medium 55/hard 30/easy 4）。
