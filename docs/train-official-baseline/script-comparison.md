# 官方入口、官方四卡配置与 DSH 配置对照

核对时间：2026-10-01。这里只比较源码、配置和已保存证据；本轮仍在资源恢复评估，没有启动新服务器，也没有本轮官方 `fit` / GPU 更新成功证据。

## 直接执行

### 脚本在哪里，是否走同一入口

三条路径属于同一 Git 仓库，但保存在两个独立 worktree。官方入口 `scripts/code/train.sh` 和底层 `recipes/code/run_train.sh` 在两个 worktree 都存在；官方四卡准备器位于 `train-official-baseline/docs/`，DSH wrapper 位于 `train-p0-integration/scripts/code/`。官方 WT 没有 DSH wrapper。

```text
官方默认：scripts/code/train.sh
          → recipes/code/run_train.sh → verl.trainer.main_ppo

官方四卡：docs/train-official-baseline/prepare_baseline.py 生成 launch.sh
          → 原版 scripts/code/train.sh → 原版 recipes/code/run_train.sh
          → 官方 a2ad9f61 的 verl.trainer.main_ppo

DSH 四卡：scripts/code/train-dsh-separate-async.sh
          → train-dsh-minimal.sh → scripts/code/train.sh
          → 集成版 recipes/code/run_train.sh → 集成版 verl.trainer.main_ppo
```

入口证据：`O:106`、`R:194`、`B:207`、`D:119`、`M:92`。共用入口文件不等于配置、harness 和训练核心完全相同。

官方来源：[XiaomiMiMo/verl README](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/README.md)、[官方 train.sh](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/scripts/code/train.sh)、[官方 run_train.sh](https://github.com/XiaomiMiMo/verl/blob/a2ad9f6160b03ff2d47e59832bfb6b289f37c917/recipes/code/run_train.sh)。README 的 9B 是 SFT 起点，本项目在此基础上做 Code RL，不是再次执行 SFT。

### 证据路径与行号约定

下表 `O:49` 等均表示该文件的具体行号。官方原版行号固定于 `a2ad9f6160b03ff2d47e59832bfb6b289f37c917`；本项目准备器及历史 evidence 固定于官方 WT 已提交 `87bffb24d4ddb4d876b0924f48d159597ca5d5e4`；DSH 文件固定于 integration WT 已提交 `c3979814267e7de9551ac717a7dee15b306d3fdc`。本轮另加脚本注释会改变工作区行号，可用 `git show <上述 SHA>:<路径>` 对照本表。

| 代号 | 文件路径 |
|---|---|
| O | [官方 WT scripts/code/train.sh](/Users/gumpm5/Documents/Code/xDAN-Train-Performance-Verl/.Codex/worktrees/train-official-baseline/scripts/code/train.sh:1) |
| R | [官方 WT recipes/code/run_train.sh](/Users/gumpm5/Documents/Code/xDAN-Train-Performance-Verl/.Codex/worktrees/train-official-baseline/recipes/code/run_train.sh:1) |
| B | [官方 WT docs/train-official-baseline/prepare_baseline.py](/Users/gumpm5/Documents/Code/xDAN-Train-Performance-Verl/.Codex/worktrees/train-official-baseline/docs/train-official-baseline/prepare_baseline.py:1) |
| L | [官方 WT docs/train-official-baseline/evidence/launch.sh](/Users/gumpm5/Documents/Code/xDAN-Train-Performance-Verl/.Codex/worktrees/train-official-baseline/docs/train-official-baseline/evidence/launch.sh:1) |
| D | [integration WT scripts/code/train-dsh-separate-async.sh](/Users/gumpm5/Documents/Code/xDAN-Train-Performance-Verl/.Codex/worktrees/train-p0-integration/scripts/code/train-dsh-separate-async.sh:1) |
| M | [integration WT scripts/code/train-dsh-minimal.sh](/Users/gumpm5/Documents/Code/xDAN-Train-Performance-Verl/.Codex/worktrees/train-p0-integration/scripts/code/train-dsh-minimal.sh:1) |
| C | [integration WT 历史 fresh-training/run/launch.sh.txt](/Users/gumpm5/Documents/Code/xDAN-Train-Performance-Verl/.Codex/worktrees/train-p0-integration/docs/train-p0-integration/operations-recoveredpod-20260930/fresh-training/run/launch.sh.txt:1) |
| H | [integration WT config/agent/code/dsh-sdk-modal.yaml](/Users/gumpm5/Documents/Code/xDAN-Train-Performance-Verl/.Codex/worktrees/train-p0-integration/config/agent/code/dsh-sdk-modal.yaml:1) |

第一列配置以用户推荐的顶层 `train.sh` 为准；括号注明直接调用底层 `run_train.sh` 时不同的默认值。DSH 列区分 wrapper 默认与历史调用者 `C`，不把旧运行参数当作新运行事实。

### 逐参数三列对照

| 参数 / 行为 | 官方 train.sh → run_train.sh 默认 | 本项目官方四卡 wrapper | DSH train-dsh-separate-async.sh |
|---|---|---|---|
| GPU 总量 / 资源池 | `8×8=64` 训练池 GPU；独立 rollout 节点数 `0`。底层单独调用为 `4×8=32`。`O:49`、`O:51`、`R:54` | `1×4=4`，训练与 rollout 共用四卡；独立 rollout 节点数 `0`。`L:24`、`L:26` | 训练池 `1×2` + 独立采样池 `1×2`，共四卡；训练池仍有 hybrid SGLang。`D:44`、`D:45`、`D:46` |
| trainer 模式 | `colocate_async`，底层 Code 默认也相同。`R:281` | 原版 `colocate_async`。`L:23` | `separate_async`，配置两次固定且最终覆盖调用者。`D:44`、`D:62`、`D:119` |
| actor TP / PP / CP / EP | `8 / 1 / 1 / 1`；底层单独调用 CP 为 `2`。`O:53`、`O:55`、`R:60` | `4 / 1 / 1 / 1`。`L:28`、`L:29`、`L:30`、`L:31` | `2 / 1 / 1 / 1`。`D:47`、`D:64` |
| rollout TP | `2`；底层单独调用 `4`。`O:57`、`R:68` | `2`；四卡训练池可分成两个 TP2 hybrid replica。`L:32`；官方 `verl/workers/rollout/llm_server.py:535` | `2`；另有独立 TP2 replica，训练池 TP2 hybrid 也会初始化。`D:69`；官方 `verl/trainer/ppo/v1/trainer_separate_async.py:86` |
| 模型输入 | `MODEL_PATH` 必填，脚本不固定 9B；README 推荐 MiMo-V2.6-Distill-Qwen-9B。`O:6`；官方 `README.md:17` | `/workspace/models/MiMo-V2.6-Distill-Qwen-9B`。`L:9` | `MODEL_PATH` 必填；历史调用者使用相同 9B SFT 路径。`D:8`；`C:8` |
| 数据输入 | `TRAIN_DATA`、`VAL_DATA` 必填。Code parquet 需 `extra_info.instance_json`、`docker_image`。`O:7`、`O:8`；官方 `recipes/code/dataset.py:35` | 固定 train8 parquet + holdout1 parquet。`L:10`、`L:11` | 路径由调用者提供；历史 train8/holdout 不改变 wrapper 的必填约束。`D:9`、`D:10`；`C:9`、`C:10` |
| harness | 四原生 whitebox：mini-mimocode / mini-bash / mini-claude-code / mini-codex。`O:11`；官方 `config/agent/code/mix-four-whitebox.yaml:2` | 同四原生 arm；从官方 `git show` 读取 profile，不接 DSH。`B:23`、`B:38`、`L:76` | 调用者必须显式提供 mix；本项目 mix 为 DSH SDK + mini-mimocode。`D:13`；integration `config/agent/code/mix-dsh-mimocode.yaml:2` |
| harness 分配 | `step-hash`，seed `20260911`；同一 prompt 的整组 N 用同一 arm。`O:12`、`O:13`；integration `recipes/code/mimoagent_runner.py:127` | 同 `step-hash` / seed；N4 的一组仍只选一个 arm，并非一组四种各一条。`L:53`、`L:54` | `paired-subgroup`，同一 prompt 的 N4 固定 DSH2 + MiMo2。`D:53`、`D:106`；integration `recipes/code/mimoagent_runner.py:125` |
| 环境后端 / 容量 | 原 profile Kubernetes；mini-mimocode CPU request `.5`、limit `4`、memory limit `8Gi`。官方 `config/agent/code/mini-mimocode.yaml:21` | 四 profile 仅环境改为 Modal，`/testbed`、CPU2、8192MB、sandbox7200s、固定 registry secret、transport error 抛出；agent / 评分等沿原版。`B:56`、`B:69` | 两 profile Modal，相同 CPU2 / 8192MB / 7200s；额外 `git_leak_prevention: strip`，集成环境 wrapper 会剥离历史。`H:13`、`H:20`；integration `recipes/code/code_environment.py:29` |
| 模型总上下文 | `MAXLEN=262144`。`O:20` | `65536`。`L:33` | `65536`。`D:49` |
| prompt / response 上限 | `16384 / 245760`（response 默认 MAXLEN − PROMPT）。`O:21`、`R:53` | 显式 `4096 / 61440`，data / rollout / PPO token 上限绑定同一配置。`L:34`、`L:35`、`L:36` | 显式 `4096 / 61440`，重复固定 data / rollout / PPO。`D:49`、`D:93`、`D:95` |
| SGLang context | 未显式传 `context_length`，传 rollout `max_model_len`。`R:235` | 显式 `context_length=65536`。`L:110` | 显式 `context_length=65536`，另禁用 custom all-reduce。`D:76`、`D:77` |
| 单次模型输出上限 | `32768`；顶层复制默认 profile 并注入该值，codex 用 `max_output_tokens`。`O:22`、`O:82` | 保留官方单 turn `32768`，自定义 mix 不走默认复制分支，所以生成器复现同样注入。`L:70`、`B:78`、`B:87` | profile `max_tokens=4096`，DSH 同时显式 `context_window=65536`。4096 是模型请求输出限制，不是已证明的整场 64K 输出预算。`M:48`、`H:8`、`H:9`、`H:29` |
| request / trajectory / agent 限制 | request3600s、retry0、trajectory4800s；mini-mimocode step_limit500。`O:23`、`O:24`、`O:72`；官方 profile `:16` | request3600s / retry0 / trajectory4800s，保留原生各 arm agent 限制。`L:71`、`L:72`；继承 `O:72` | profile request600s / retry0，DSH `run_timeout=1800s`；wrapper trajectory 默认3600s，历史 C 改4800s；MiMo step500。`H:7`、`H:30`、`M:64`、`C:49`；integration `mini-mimocode-modal.yaml:16` |
| N / global / mini / micro batch | `16 / 64 / 64 / 1`；底层单独调用 `8 / 32 / 32 / 1`。`O:15`、`O:16`、`O:17`、`O:37`、`R:46` | `4 / 1 / 1 / 1`。`L:37`、`L:38`、`L:39`、`L:40` | `4 / 1 / 1 / 1`，最终固定。`D:48`、`D:92` |
| batching / token budget | actor dynamic `True`；PPO token 默认 MAXLEN / CP。`O:25`、`R:66` | actor / rollout-logprob / ref-logprob static；PPO65536。原版 validator 可接受这些配置。`L:41`、`L:36`、`L:111`、`L:112` | actor / rollout-logprob static；PPO65536。`D:79`、`D:80`、`D:96` |
| 算法 / loss / STD | GRPO；prompt-mean；不除组 STD，`norm_adv_by_std=False`。官方 `recipes/code/config/train.yaml:153`；`O:30`、`O:31` | 同官方 GRPO / prompt-mean / STD=False。`L:56`、`L:57` | 仍为 GRPO / prompt-mean / STD=False，底层数学函数未改；但分组方式改变，不能说整体算法语义完全相同。继承 `O:30`、`O:31`；`D:91` |
| GRPO 按 harness 分组 | `group_advantage_by_harness=False`；整组同 arm。`R:324` | `False`。`L:55` | `True`，DSH2 与 MiMo2 分别中心化，不跨 arm 混算优势。`D:91`、`D:103` |
| 动态过滤 / 采样 admission | 官方 `filter_groups=True`，metric reward；内置 ReplayBufferAsync。`O:32`、`O:33`；官方 `verl/trainer/ppo/v1/trainer_base.py:416` | 同官方 reward group filtering。`L:58`、`L:59` | wrapper 默认关闭原版 filter；改用 `CompleteCodeGroupAsyncSampler`，要求完整4条且2+2 arm counts，可补采，wait5400s、pending2。`M:50`、`D:100`、`D:106`、`D:108` |
| 学习率 / weight decay | `1e-6 / .01`。`R:228`、`R:229` | 沿原版，未额外覆盖。`L:109` | 沿原版，wrapper 未固定其他 LR。`D:119`、`M:92` |
| sampling / entropy | temperature1 / top_p.95 / top_k20；entropy coeff0，仍计算 entropy；chunk16384。`O:26`、`O:34`、`O:36`、`R:213` | 同原版。`L:60`、`L:62`；继承 `O:26` | temperature等继承；coeff0；默认 entropy chunk4096，历史 C 可改。`D:58`、`M:49` |
| off-policy / 权重同步 | threshold2 / drop，warmup1；colocated 同步后继续生成。`O:38`、`R:282`；官方 `trainer_colocate_async.py:48` | 同2/drop/warmup1。`L:63`、`L:64` | 同2/drop；separate parameter_sync_step1 / warmup1，standalone NCCL 权重同步128MB bucket。`D:71`、`D:73`、`D:74` |
| weights / grads / Adam 精度 | 训练 dtype BF16；非 precision-aware optimizer，配置 main_grads / Adam m / v FP32；不能称全部权重 FP32。官方 `verl/trainer/config/engine/megatron.yaml:122`；`verl/trainer/config/optim/megatron.yaml:54`、`:58`、`:61`、`:64` | 保留原版精度配置，不启用我们的 optimizer / warmup 补丁；实际 GPU dtype 仍待本轮初始化验收。`L:109` | precision-aware=True，main grads / Adam m / v FP32，显式 DDP FP32 reduce。`D:83`、`D:84`、`D:85`、`D:86`、`D:87` |
| phase offload / CPU optimizer | phase param / optimizer / grad offload=True。不等同 CPU Adam compute。`O:58`、`R:258` | 同原版 phase offload=True。`L:42` | phase offload 默认True；单独固定 CPU optimizer offload=False、fraction0。`M:37`、`D:88`、`D:89` |
| SG KV / prefill / mem | `fp8_e4m3` 是原版 run_train 默认，flashinfer；prefill32768，GPUmem.78；底层单独默认.75。`R:96`、`R:98`、`O:59` | KV/prefill/mem 保留官方默认；FP8 KV 不是 wrapper 新增的训练降精度。继承 `R:96`、`R:98`、`O:59` | 同原版 FP8 KV；prefill4096，GPUmem.60，Mamba cache16。`M:38`、`M:61`、`M:63` |
| agent worker / gateway / sessions / running requests | `64 / 8 / 2048 / 128`。`O:67`、`O:68`、`O:69`、`O:71` | `1 / 1 / 4 / 4`。`L:49`、`L:50`、`L:51`、`L:52` | `1 / 1 / 4 / 4`。`D:51`、`D:52` |
| steps / epochs | 200 / 10。`O:18`、`O:19` | fresh 总step1；resume 从本次CP1到总step2；epochs10。不是原论文规模复现。`L:82`、`L:83`、`L:45` | TOTAL_STEPS 必须调用者提供；epochs默认2。历史 C 总step4 / epochs3。`D:18`、`M:45`、`C:55`、`C:56` |
| checkpoint | SAVE5、默认同步，model / optimizer / extra；不限制保留个数。`O:79`、`R:297`；官方 `verl/trainer/config/actor/actor.yaml:135` | SAVE1、同步、额外保存 HF model；model / HF / optimizer / extra，完整CP验收后才允许恢复。`L:43`、`L:113`、`L:114` | wrapper默认 SAVE1、默认原版 contents；历史 C 改SAVE4 / keep2 / 同步 / HF。`M:54`、`C:57`、`C:82`、`C:83`、`C:90` |
| 恢复入口 | 默认 resume disable、resume_path=null；底层支持 resume_path。`R:293`、`O:110`；官方 `verl/trainer/ppo/v1/trainer_base.py:1065` | 原脚本 `trainer.resume_mode=resume_path`、本次 `CP/global_step_1`；WB新IDa9off002、独立resume目录。`L:80`、`L:83`、`L:84` | wrapper只限制 legacy sync checkpoint opt-in，未固定本次恢复路径；历史 C 为 fresh disable / resume_dataloader=False，不能当成恢复配置。`D:110`、`C:87`、`C:88` |
| holdout / test | VAL_N1 / batch500 / deterministic；TEST-1，但最后step会验证。`O:42`、`O:43`、`O:80`；官方 `trainer_base.py:727` | VAL_N1 / batch1 / deterministic、TEST2；phase1 step1 是最后step，也会触发 holdout，不应只描述phase2才验证。`L:44`、`L:46`、`L:47`；官方 `trainer_base.py:727` | VAL_N2 / batch1 / deterministic、默认TEST-1，最后step规则仍适用。`D:55`、`M:55` |
| 日志 / W&B / Insight | 默认 console / tensorboard；源码原生另支持 file / wandb / rl_insight。`R:287`；官方 `verl/utils/tracking.py:48` | 五logger，file绑定本RUN metrics.jsonl，WBa9off001/a9off002；Insight18080、enable1；Ray转发路径及run元数据。`L:15`、`L:16`、`L:82`、`L:88`、`L:106`、`L:116` | wrapper默认 console / tensorboard / file / wandb，可由TRAINER_LOGGERS覆盖；历史 C 加rl_insight。file路径显式Ray转发。`D:111`、`D:116`、`C:71` |
| secret / logs / runtime 归属 | 原版 RUN_DIR 派生TB/dumps/trajectories/rollouts/validation/cfg；runtime PYTHONPATH。`R:121`、`R:168` | 显式本次RUN及resume子目录，NETRC / MODAL_CONFIG_PATH仅转发文件路径，源a2ad在PYTHONPATH首位；缓存rg原路径复用。`L:12`、`L:13`、`L:17`、`L:75`、`L:87`、`L:106` | RUN_DIR / W&B ID必填；source身份由调用者冻结；runtime按集成脚本转发所需路径。`D:11`、`D:17`、`D:42`；`C:44` |

### 三种 trainer 模式与最低卡数边界

官方通用 `ppo_trainer.yaml:235` 明确列出三种模式，`ppo_trainer.yaml:236` 通用默认是 `sync`；Code recipe `recipes/code/config/train.yaml:207` 覆盖为 `colocate_async`。官方 `verl/trainer/ppo/v1/__init__.py:17` / `:18` / `:19` 导入三个注册实现；`verl/trainer/main_ppo.py:142` 按配置查注册类并执行 `init`、`fit`（`:154`、`:156`）。这是底层能力，不代表原版 Code shell 预检同时允许三种模式。

| 维度 | sync | colocate_async | separate_async |
|---|---|---|---|
| 官方实际注册 | `trainer_sync.py:24` | `trainer_colocate_async.py:25` | `trainer_separate_async.py:39` |
| 资源池 | train / rollout 同池。`trainer_sync.py:27` | train / rollout 同池。`trainer_colocate_async.py:28` | 训练池 + 额外standalone池；训练池hybrid仍先初始化。`trainer_separate_async.py:55`、`:82`、`:86` |
| partial rollout | 关闭。`trainer_sync.py:28` | 开启；采样结束abort/sleep，更新后resume。`trainer_colocate_async.py:29`、`:55`、`:48` | 开启；standalone持续采样，训练池SG切换/验证仍存在。`trainer_separate_async.py:43`、`:146`、`:187` |
| 训练池“借给采样” | 正常轮流共享。`trainer_sync.py:40` | 正常轮流共享。`trainer_colocate_async.py:55` | 初始化时加入采样balancer、训练前移除；验证可切回。空闲自动切回的 `should_switch_to_rollout()` 原版直接返回False，不能承诺持续自动借池。`trainer_separate_async.py:99`、`:194`、`:201`、`:205` |
| 原版 Code validator | 不接受 | 接受 | 不接受；integration已增加模式参数 |
| 当前已准备脚本 | 没有本次sync launcher | 官方wrapper固定4卡 / actorTP4 / rolloutTP2 | DSHwrapper固定2训练+2独立采样 / 两边TP2 |

validator证据：官方 `recipes/code/validate_resolved_config.py:72` 写死 `colocate_async`，integration同文件改为接受调用者 trainer mode。因此不能只改原版 `TRAINER_MODE` 就宣称原版 Code 同时通过三种预检。原生单harness也有独立限制：原版 `recipes/code/mimoagent_runner.py:88` 要求mix至少两项，尽管函数注释提到one-armed spec；integration已经放宽该检查，不能把集成版能力归为原版。

**卡数要分拓扑下限和9B/64K真实容量。** 官方资源池大小由节点数×每节点GPU数构造（`trainer_base.py:1018`），SG replica数量由池world_size / rollout_world_size计算（`llm_server.py:530`、`:535`），没有独立于TP/模型的“Code训练固定至少4卡”结论。本次两个脚本均明确固定四卡：官方actorTP4直接需要四卡；DSH保持训练TP2、独立rolloutTP2也直接需要2+2。两卡colocate、2+1 separate是改TP及容量后才能评估的候选，不是已有实测；单卡rollout也意味着改当前TP2。不能从注册模式或权重字节数推断9B/64K、FP32训练states、通信buffers与KV全部能稳定装入。

### 核心代码、提交和本轮证据

| 项目 | 官方四卡路径 | DSH路径 |
|---|---|---|
| 运行源码身份 | B固定官方a2ad9f61 + 原版两子模块pin，archive从Git对象生成，不把本地WT修改混入。`B:18`、`B:19`、`B:221` | 集成源码有独立commits及不可变source；旧C固定source829，后续b59/最新配置必须各自验收，不能只换source字段冒充旧receipt通过。`C:44`；integration `tasks/train-p0-integration/handoff.md:8` |
| trainer / runner / optimizer核心 | 运行archive没有我们算法、runner、optimizer补丁。官方WT为push门禁曾有9个Python文件格式修复+2notebook整理，与GPU部署源码身份分开。官方 `docs/train-official-baseline/README.md:38` | 相对a2ad，改了mimoagent_runner、run_train、validator、trainer_base、engine配置、transformer实现；新增DSH agent/runner/proxy、code_environment、complete_group_sampler、grad_sync_warmup。不是仅改环境变量。例：integration `mimoagent_runner.py:314`、`trainer_base.py:420`、`transformer_impl.py:473`、`:680`、`:1032` |
| DSH真正入口 | 无DSH | `DshSdkAgent` 调本项目 Python SDK runner；runner构造 `DeepSeekHarnessConfig` 后 `DeepSeekHarness(...).run(...)`，不是upstream黑盒DSH CLI。token/logprob训练数据走原生Gateway/TransferQueue。integration `recipes/code/dsh_runner.py:84`、`:97`；`mimoagent_runner.py:330`；`docs/train-p0-integration/original-entry-four-gpu-64k-dsh.md:19` |
| 本地分支 / HEAD | `worktree-train-official-baseline` / `87bffb24`；本地远端跟踪ref同SHA，此前官方方案已commit/push。官方handoff及 `README.md:82` | `worktree-train-p0-integration` / `c3979814`；此前保存已commit、未push，全库门禁阻挡；旧未跟踪operations文件另存。integration `tasks/train-p0-integration/handoff.md:7` |
| 本轮新增注释 / 本文 | 注释/本文随本轮提交保存，不进入冻结的GPU运行源码；远程推送以本轮实际回执为准 | 注释已独立本地commit `f899fe966385830e9b64d380e889bb0ad376b13c`；仍未push，历史全库门禁未过；本文只读核对 |
| 实际运行证据 | 旧实例CPU配置预检returncode0；官方更新0、acceptanceFalse，未证明GPU init/fit/checkpoint/resume。官方 `docs/train-official-baseline/evidence/pause-status.json:8`、`:9`、`:15` | b59完整9B init-only通过但0更新，后续DSH65536/1800s最新配置未fit；更早旧run更新不计入本轮。integration `tasks/train-p0-integration/handoff.md:3`、`:8` |
| 源码部署留证 | 旧实例receipt核对官方SHA、1810regular files +6symlinks；旧Pod后来not_found，不代表新服务器已恢复。官方 `docs/train-official-baseline/evidence/deployment-receipt.json:3`、`:7`、`:8`；`pause-status.json:5` | 历史源码/环境/运行留证属于旧实例，恢复后的uv、GPU、Ray、服务和认证要重新证明；不沿用旧活跃状态。integration handoff `:9` |

本次核对只读取两WT并新增本文。HEAD / branch / status由本地Git实查；push状态结合本地remote-tracking ref与已提交handoff记录，不声称已进行新的远程push验证。

### 四卡 colocate 与 2+2：速度、容量和质量

这里是锁定源码的结构分析，不是已经完成的 GPU 对照测速。本轮官方更新数仍为0；此前90%采样耗时也不能作为恢复后的新run实测。

| 维度 | 四卡 colocate_async | 2+2 separate_async |
|---|---|---|
| 采样阶段 | 四卡，两个TP2 replica；能否增加吞吐取决于请求并发和GPU是否饱和 | 稳态独立采样池两卡，一个TP2 replica；初始化/验证使用hybrid池，不能计入自动持续借卡 |
| 参数更新 | 四卡TP4，状态分摊更宽裕；TP通信使训练速度不保证优于TP2 | 两卡TP2，状态/activation余量更紧；可与独立采样重叠 |
| 等待/同步 | 采样暂停、更新、同步与恢复，有阶段切换成本；partial rollout仍保留异步能力 | 可隐藏部分训练等待，另有权重传输、版本滞后与陈旧样本丢弃成本 |
| 64K容量 | 更多训练卡有利于容量，但实际峰值、optimizer第一步与长轨迹仍待验收 | 两训练卡容量更紧，不能用旧init-only证明64K真实fit |
| 模型质量 | 共享池拓扑本身不提升任务能力；partial rollout也需版本控制 | 独立池拓扑本身也不提升任务能力；需检查版本滞后/丢弃及有效训练数据 |
| 成本比较 | 同一四卡Pod每小时价格相同，应比较有效训练样本/美元 | 同左；token/s更高也可能被丢弃率抵消 |

仅作直观估算：忽略启动、工具等待重叠、验证/保存、过滤和各种同步细节，colocate一周期约为 `R_colocate + T_TP4 + S_switch`，separate稳定流水约为 `max(R_separate, T_TP2) + S_sync`。两侧R和T不是同一个数，不能把相同硬件的两个公式直接代入同一时长宣称异步提速。

**条件示例：**若colocate的采样占周期90%，并且采样吞吐完全不变、其余10%全被隐藏，理想加速为 `1 / 0.9 ≈ 1.11×`（耗时下降10%）。但2+2将常驻采样池从两个TP2 replica减到一个。若此前采样充分利用两个replica、工作可线性并行且工具等待不是主因，则R可能接近翻倍，反而抵消重叠收益。这个条件估算不是RTX6000实测，不能承诺倍速或必然减速。

90%“采样时间”还可能含Modal启动、工具执行、排队或超时。需分解GPU decode/prefill与环境等待，再决定增采样卡、提高并发还是调整任务后端。当前batch1/N4/worker1的小闭环尤其不能代表高并发稳态吞吐。

公平对照应固定模型/数据/harness/分组/输出上限/过滤/精度，只改拓扑；当前官方wrapper与DSH wrapper同时改变这些因素，不能将其reward差异归因于colocate或separate。完成闭环后再记录有效组/小时、accepted tokens/s、各阶段耗时、四卡峰值显存、权重版本滞后、丢弃/截断率和同一holdout成功率。

## 深度交互

“官方完整跑通”宜先定义为真实更新 → 完整checkpoint → 从该checkpoint恢复 → 恢复后新更新 → holdout，而不是脚本解析成功。当前官方四卡方案沿用官方训练核心和四原生harness，但N4/batch1/两步只验证闭环；step-hash的一两组也不能保证四个arm都实际运行。若目标是所有原生arm覆盖，应读取实际trajectory的arm记录后单独验收，不能因mix中列出四项就宣布覆盖。

64K至少有三个不同值：训练/rollout模型context、每次请求输出上限、整条agent轨迹实际长度。官方四卡是65536 context +32768单次输出；DSH是65536 context +4096单次输出 +1800s SDK时限。两者都不等于已经执行过64K长轨迹，需以真实token、截断原因及传回训练的trajectory统计证明。

恢复评估应先选择官方colocate四卡闭环，再依据实测峰值与吞吐评估缩卡。把DSH切回原版wrapper不只是换harness：会同时改变分组、过滤、sampler、precision-aware optimizer和核心补丁。保留上述三列配置可避免将容量调整、官方兼容性验证与DSH算法扩展混为同一次比较。

旧脚本硬截止 `2026-09-30T12:51:43.743Z` 已过期，旧RUN / W&B ID / `/opt` uv路径与旧Pod身份均属历史绑定（`B:26`、`B:29`、`B:30`）。恢复时需有新的授权期限及RUN身份、相应守护和真实环境验收；本文不是新启动指令，也不延长原预算。

脚本注释最有价值的内容是入口调用链、每个非默认参数的目的、哪些是历史run绑定，以及“64K能力/CPU解析/真实fit”分别需要什么证据。避免在注释中写“已跑通四卡64K”或给两卡候选贴上实测标签。
