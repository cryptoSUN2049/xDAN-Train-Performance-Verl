# xdan-verl-fusion 交接（M1：verl 融合分支）

## 1. TL;DR

- 分支 `xdan/fusion-a9f2985`：上游 verl `a9f29851` + MiMo `e2b9fc03`，**只在本地，未 push**。
- 冲突 7 个文件，全部已解决。取舍说明见 `docs/xdan-verl-fusion/conflict-resolution.md`。
- CPU 验证：
  - MiMo 新增或改动的测试 326 个，全部通过（不含需要 triton 的 1 个）；
  - 上游 trainer/ppo、workers/config、agent_loop 子集 449 个通过，另有 1 个因缺本地模型目录失败（环境原因）；
  - 5 个 recipe 主配置都能完成 Hydra 组装。
- 委托方：xDAN-DSH-uni-agent 会话 `verl-uni-agent-harbor-opd-rl-a2`，任务书见 `/Users/gumpm5/Documents/Code/xDAN-DSH-uni-agent/.claude/worktrees/fusion-mimo-uni-agent/docs/fusion-mimo-uni-agent/m1-verl-fusion-brief.md`。
- 下一步：用户确认后 push；M1b（挑入 P0 线的 verl 修复）由谁执行待用户定。

## 2. 本轮交付物

| 提交 | 内容 |
|---|---|
| `558187e1` | cherry-pick `e2b9fc03`。冲突块先取上游一侧（中间态，不保证能通过 MiMo 测试） |
| `6c6bd939` | 冲突解决，共 7 个冲突文件，另加 `trainer_separate_async.py` 的签名跟改，以及 `docs/xdan-verl-fusion/conflict-resolution.md` |
| `269224d9` | 修复 MiMo 两个 DAPO 测试 fixture（这两个用例在 pristine mimo-oss 上本来就失败） |
| （本提交） | 本交接文档 |

- `_generated_*.yaml`：重新生成后与自动合并结果完全一致，脚本输出 `All good`，所以没有单独的生成提交。

## 3. 设计约束

- 分支形态固定为"上游 + 少量独立提交"。上游升级时整段 rebase，不混改。
- verl 侧的改动只进 `xdan/fusion-*` 分支，不进 `mimo-oss` / `main`。
- 委托方要求**不要在 `mimo-oss` / `main` 上重排 `e2b9fc03`**，避免他们的基线漂移。
- xDAN 的 verl 补丁（teacher padding、dense entropy、finish_reason）由委托方另起提交，本轮不带。

## 4. 已踩坑和已确认的真实行为

- 在 worktree 隔离会话里，复合 git 命令（含 `$(...)`、for 循环、`cd &&`）会被 harness 拒绝，要拆成单条命令执行。
- 测试环境用本 worktree 的 `.venv`（Python 3.12，torch 2.11 CPU，transformers 5.9.0）。
  - TransferQueue 要固定为 **0.1.9**，与 P0 线生产锁一致；git main 是 0.1.12.dev0。
  - 两个版本下，那 2 个 MiMo fixture 用例都会失败，与 TQ 版本无关。
- `third_party/mimoagent-osr` 必须 `git submodule update --init`，再 `uv pip install -e`，否则 3 个 design 测试在收集阶段就报错。
- `tests/utils/megatron/test_muon_layerwise_bridge_ddp_on_cpu.py` 需要 triton，mac 上没有对应 wheel，只能在 Linux 上跑。
- `tests/workers/config/test_model_config_on_cpu.py::test_target_modules_raises_on_invalid_type` 需要 `~/models/Qwen/Qwen2.5-0.5B`，属于环境问题。
- lint：
  - 本分支改过的文件，ruff 0.12.2（pre-commit 钉的版本）和 0.15.8 都通过；
  - 全仓 `ruff format --check` 报的 9 个 MiMo 文件是 e2b9fc03 原样带来的，也就是我们 main 上 `fc6582e9` 格式化过的那批，本分支没有处理。
- 融合后组装出的配置与 mimo-oss 对比，被移除的键有：
  - `actor_rollout_ref.{actor,ref}.router_replay`
  - `{ref,critic}.megatron.grad_offload`
  - `data.continuous_token`

  这些是上游的变更。MiMo recipe、run 脚本和 P0 线脚本都没有引用它们。

## 5. 下一里程碑清单

- [ ] 用户确认后 `git push origin xdan/fusion-a9f2985`
- [ ] 回报委托方：分支名、SHA、文档路径、测试结果
- [ ] M1b：按依赖顺序挑入 P0 线的 verl 修复。`trainer_base.py` 和 `transformer_impl.py` 按 conflict-resolution.md 的取舍重放。执行方由用户决定。
- [ ] 可选：挑入 `fc6582e9` 的格式化（独立提交）

### M1b 候选提交（`mimo-oss..worktree-train-p0-integration -- verl`，按时间从早到晚）

表中 TB 指 `trainer_base.py`，TI 指 `transformer_impl.py`。

| SHA | 功能 | 验证状态 | 是否碰 TB/TI |
|---|---|---|---|
| `fc6582e9` | 只改 lint（length_penalty） | 不涉及 | – |
| `0bfd0630` | Megatron checkpoint 恢复顺序 | 包含它的源码做过 GPU resume；它针对的 CPU-optimizer 场景只有 CPU 测试 | – |
| `91d7e956` | sync 最后一个 batch 的 dispatch | 包含它的源码在 GPU 上跑过（64K r2） | TB |
| `f0f3fc3f` | 恢复时不加载梯度（`load_grad=False`） | GPU 验证过（恢复后 step3 grad_norm=0.69） | TI |
| `2886da5c` | TE optimizer 恢复输入先放 CPU | GPU 小规模测试结果逐位一致；**只在 TE 2.16.1 + torch 2.11 下启用，换了上游版本后可能不生效** | – |
| `ab95dab3` | 保留 rollout identity，供覆盖率审计 | 包含它的源码在 GPU 上跑过 | TB |
| `a910dad6` | validation 读取扁平化的 reward metadata | **只有 CPU 测试** | TB |
| `52ea0d4e` | resume 时重置 sync dataloader | 包含它的源码在 GPU 上跑过 | TB |
| `5b0a83f0` | entropy 系数为 0 时不保留 autograd | GPU 验证过（TP2 探针，16 个场景） | TB+TI |
| `0fab9666` | separate_async 只采完整组的 profile | 包含它的源码在 GPU 上跑过（4 卡 async） | TB |
| `ced8e69d` | async 恢复前对旧 loader 重置加 gate | 包含它的源码在 GPU 上跑过（4 卡 async） | TB |
| `e4c3dacb` | 预热梯度同步的 communicator | 原版在 GPU 上失败，由下一个提交修正；**必须和下一个一起取** | TI |
| `b59d76d7` | NCCL 多后端 | 只在 GPU 上验证到初始化；从未在开启 warmup 的情况下跑过训练 update；`production_grad_sync_warmup_verified: false` | – |

建议的最小集合：

- 先取：`0bfd0630`、`f0f3fc3f`、`2886da5c`、`5b0a83f0`，加上 dataloader-resume 相关的 `52ea0d4e`、`ced8e69d`。
- 暂缓：`e4c3dacb` 和 `b59d76d7`，等 GPU 复现"resume 后第一次 update"之后，再带上验证证据单独合入。
- 单独评审：`a910dad6`、`0fab9666`。
- 详细证据路径见本会话发给委托方的回报。
- [ ] GPU 复核：MiMo Music 2 卡配置在融合分支上跑单步 smoke，同时核对 DAPO 过滤是否与 mimo-oss 一致（`rm_scores` 优先于 `reward_extra_info["reward"]`）

## 6. 分支和部署状态

- worktree：`.claude/worktrees/xdan-verl-fusion`。本地另有一个由 EnterWorktree 创建的空分支 `worktree-xdan-verl-fusion`，可以删除。
- 分支 `xdan/fusion-a9f2985` 只在本地，未 push，没有 CI。
- 没有动 `main`、`mimo-oss`，也没有动其他 worktree 分支。没有使用 GPU 或 RunPod。

## 7. 冷启动 checklist

1. 读本文件，再读 `docs/xdan-verl-fusion/conflict-resolution.md`。
2. `git log --oneline a9f29851..xdan/fusion-a9f2985`，确认提交结构。
3. 复现测试：在 `.venv` 中执行 `python -m pytest -q -o addopts="" tests/trainer/ppo tests/workers/config tests/recipes tests/experimental/fully_async_policy/test_mimo_grouping_on_cpu.py`。
4. Hydra 组装：`python -m verl.trainer.main_ppo --config-path=$PWD/recipes/<arm>/config --config-name=<name> --cfg job`。
