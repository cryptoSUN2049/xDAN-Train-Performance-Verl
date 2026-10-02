# xdan/fusion-a9f2985 冲突取舍说明

- 分支：`xdan/fusion-a9f2985` = 上游 verl `a9f29851` + cherry-pick MiMo `e2b9fc03`
- 合并基：`git merge-base a9f29851 e2b9fc03` = `2781c1e4`
- 提交结构：
  1. cherry-pick 提交：7 个冲突文件里**每个冲突块都先取上游一侧**，其余非冲突改动照原样带入。这个中间提交不保证能通过 MiMo 测试。
  2. 冲突解决提交：本文件里的全部取舍，审查时只看这一个 diff 即可。
- 原则：以上游 `a9f2985` 的行为为底；MiMo 新增能力以开关或兜底的方式叠加；MiMo recipe 依赖的默认值不变。

## 总表

| 文件 | 冲突块 | 结论 |
|---|---|---|
| `pyproject.toml` | 1 | 取 MiMo 新增的 `[tool.ruff.lint.per-file-ignores]`；mypy 段的注释头按上游删除 |
| `verl/trainer/ppo/v1/replay_buffer.py` | 1 | 合并：`metric=reward` 时以上游 canonical `rm_scores` 为准，MiMo 的 infra 标记、按 harness 分组的统计全部保留 |
| `tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py` | 1 | 并集：上游的 `canonical_rewards` 和 MiMo 的 `reward_field` 两套 fixture 同时保留 |
| `verl/trainer/ppo/padding_utils.py` | 5 | 取 MiMo 的显式 `sequence_length_multiple`；调用方不传（=1）时退回上游的 `SYNTHETIC_PADDING_SEQ_LEN=128` |
| `verl/trainer/ppo/v1/trainer_base.py` | 2 | (a) `prepare_step` 新增 `prefetch_next_batch` 参数；(b) spec-decode 统计改用 MiMo 的 `extract_spec_decode_stats` |
| `verl/trainer/ppo/v1/trainer_separate_async.py` | 0（语义跟改） | `prepare_step` 签名同步透传 `prefetch_next_batch` |
| `verl/workers/engine/megatron/transformer_impl.py` | 1 | 条件取并集：保留上游"要求 remove-padding"，同时采用 MiMo 的"只有 MTP **训练**才禁用 fused kernels" |
| `tests/utils/test_check_profiler_output.py` | 1 | 取上游。MiMo 这里只是去掉了一对括号（纯格式），而上游已经重写了这段 validator 的 API |

## 逐文件说明

### 1. `verl/trainer/ppo/v1/replay_buffer.py`（`_dapo_filtered_keys`）

- **上游**（#7792 `7cb65014`）：`filter_groups.metric == "reward"` 时只读 `rm_scores`，作为 canonical 的 pre-KL reward；`ppo_trainer.yaml` 的默认 metric 也从 `null` 改成了 `reward`。
- **MiMo**：
  - 总是读取 `extra_fields`（`is_infra` 和 harness 标签都在里面）；
  - 优先读 `reward_extra_info["reward"]`，没有时才回退到 `rm_scores`；
  - 新增 infra 感知的分组判定 `_classify_group`，以及按 harness 拆分的 raw 指标。
- **取舍**：
  - 数据获取沿用 MiMo：总是取 `extra_fields`；metric 为 `reward` 时额外取 `rm_scores`。否则 MiMo 的 infra 剔除和 harness 统计会全部失效。
  - 取值优先级沿用上游：`reward` 优先用 `rm_scores`，只有当该条轨迹没有 `rm_scores` 时才回退到 `reward_extra_info["reward"]`。理由是 GRPO 计算 advantage 用的就是 `rm_scores`，DAPO 判断"组内零方差"也应该看同一个量。
  - 对 MiMo recipe 的影响：MiMo 五个 recipe 全部是 `metric: reward`。在 replay buffer 读取的时点，length penalty 还没有作用到 `rm_scores` 上，两者数值应当相同，所以预计没有行为差异。**待 GPU 复核**：如果某个 harness 写入的 `reward_extra_info["reward"]` 与 `rm_scores` 不一致，DAPO 的过滤结果会和 mimo-oss 不同。

### 2. `tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py`（`RolloutProducer.run`）

两边改的是同一段 fixture：上游加了 `canonical_rewards`，MiMo 加了 `reward_field`，两者互不冲突，所以并存，各自的测试用例都保留。

### 3. `verl/trainer/ppo/padding_utils.py`

两边修的是同一个问题：合成 padding 样本太短，Megatron 在打包 TP×CP 时会切出空片。

- **上游**（#7593 `24f25b03`）：把长度固定为 128，覆盖 TP×CP ≤ 64 的情况。
- **MiMo**：由 `get_megatron_sequence_length_multiple` 精确计算对齐倍数（CP>1 时为 `2·TP·CP`，否则为 `TP`），trainer 再显式传入 `sequence_length_multiple`。
- **取舍**：`seq_len = sequence_length_multiple if sequence_length_multiple > 1 else SYNTHETIC_PADDING_SEQ_LEN`。
  - 调用方给了精确倍数时用 MiMo 的值，这能覆盖 TP×CP > 64 的情况，样本也更短。
  - 没给时（上游的调用点，或 FSDP 得到的倍数为 1）保持上游的 128。
  - MiMo 测试断言的 32 token 对齐因此不受影响。
  - 与 mimo-oss 的唯一差别：倍数为 1 时，padding 样本从 2 token 变成 128 token。这些样本仍然被完全 loss-mask，只多一点计算量。

### 4. `verl/trainer/ppo/v1/trainer_base.py`

**(a) `step()`**

- **上游**（#7373 `890dfc3e`）：把 `_add_batch_to_generate()` 抽成可覆盖的 `prepare_step()`。`TrainerSeparateAsync.prepare_step` 在提交 batch 之后，还要等样本足够可采，并把 hybrid engine 从 rollout 切回 trainer。
- **MiMo**：新增 `prefetch_next_batch=not is_last_step`，最后一步不再预取下一批 rollout。
- **取舍**：不能在最后一步整体跳过 `prepare_step()`，否则 separate_async 不会切回 trainer。做法是给 `prepare_step` 加关键字参数 `prefetch_next_batch`，它只控制是否调用 `_add_batch_to_generate()`；子类透传这个参数，并照常执行等待和切换。

**(b) spec-decode 统计**

上游把校验逻辑内联在这里；MiMo 抽成了 `ray_trainer.extract_spec_decode_stats`，额外处理 TQ 中完全没有 `extra_fields` 的情况（外部 harness 会出现）。MiMo 的版本是上游的超集，并且有单测覆盖，因此取 MiMo。

### 5. `verl/workers/engine/megatron/transformer_impl.py`（`_maybe_enable_fused_kernels`）

- **上游**：fused kernels 要求 `use_remove_padding`。
- **MiMo**：MTP 只用于 rollout 推测解码、不参与训练时（`mtp.enable and not mtp.enable_train`），允许使用 fused kernels。
- 两边改的是两个独立的条件，因此取并集：`not use_remove_padding or is_value_model or (mtp.enable and mtp.enable_train)`。

### 6. `pyproject.toml`

- MiMo 为 prompt 原文文件新增了 E501 豁免（webdev 和 general 的工具描述、eval rubric），并把 `recipes/design/grader_service` 加入 ruff 的 exclude。这两项都取 MiMo。
- 上游删除了 mypy 段的分隔注释，按上游处理。

## 已知与本次合并无关的问题

- 在 pristine `mimo-oss` 上，`test_replay_buffer_on_cpu.py` 的以下 2 个用例本来就失败（已用 `git archive mimo-oss` 加同一个 venv 复现）：
  - `test_dapo_reward_metric_uses_rm_scores_when_reward_is_not_extra_info`：FakeRefiller 补进来的组只带 `acc`，没有 `rm_scores`；
  - `test_dapo_reports_raw_metrics_for_paired_harness_subgroups`：用 `metric=reward`，但 fixture 只写了 `acc`。

  两者都是 fixture 缺陷，修复放在后续单独的 test 提交里。
