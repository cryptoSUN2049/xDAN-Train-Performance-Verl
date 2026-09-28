# P0 / 可复现训练基线

## 范围与非目标
先从公开MiMo SFT起点完成一个领域的可信RL闭环。优先Code或General需结合可用K8s、judge与业务场景决定；暂无硬件与服务预算，不在本轮启动训练。暂不重写GRPO、加入完整Pro训练栈或一次性统一五域。

## 任务与退出条件
- [ ] **P0.1 资产锁定**：固定CUDA/PyTorch/SGLang/Megatron/Transformers组合与镜像digest，运行逐worker probe；不将通用安装脚本当专用锁文件。
- [ ] **P0.2 数据schema**：保留原始文件；解析instance_json，校验任务ID、工作目录、镜像、grader字段；转换输出可追溯。
- [ ] **P0.3 镜像mapping**：读取dataset_image→dockerhub_image；缺失或多义映射显式失败；避免General双registry前缀。
- [ ] **P0.4 环境验收**：单任务Pod创建/工具/评分/清理；区分任务失败、judge失败、infra失败。
- [ ] **P0.5 独立holdout**：按任务和来源隔离，输出split manifest；不把train复制为val；内部评测缺失明确标注。
- [ ] **P0.6 SFT基线**：同harness、同预算记录成功率/时延/token/成本；至少检查重复运行波动。
- [ ] **P0.7 短程GRPO**：真实参数更新、保存和恢复checkpoint；固定holdout前后对照。
- [ ] **P0.8 可重放记录**：run_id连接revision、resolved config、trajectory、reward、grader和资源日志。

## 未来转换器契约（尚未实现）
输入：原始parquet、image-mapping、任务根目录、固定split规则。输出：converted parquet、train/holdout manifest、缺失资产报告、输入输出hash。禁止静默丢样本、改任务语义、将错误镜像fallback成其他镜像。

## Harbor 接入决策
Harbor是环境/任务执行与评测框架，不是GRPO的替代算法。公开MiMo 9B recipe已有环境，但不是已验证的Harbor主训练pipeline。若业务需要Harbor统一环境，先做一条任务adapter，明确task schema→环境生命周期→工具/观察→verifier→trajectory映射，保持外部评分器独立，再比较同任务MiMo原路径与Harbor路径。

## Pro 方法扩展
多领域混合训练、GRS/GAR、MOPD2均需各自设计、源码实现、消融和成本验证。报告里的Pro方法不能自动写成9B公开方案已具备。Ornith式任务/scaffold生成属于后续P4研究，不在P0同时引入。

## 长期性能指标
能力：固定集成功率、跨harness泛化。系统：有效token吞吐、峰值显存、环境启动/队列等待、陈旧轨迹比例。成本：GPU-hours/CPU/judge费用与每成功任务成本。可靠性：infra/judge失败率、hacking检出、配置可追溯覆盖。

## 仍需项目输入
GPU型号与数量、可用Ray/K8s/共享存储、首个业务领域、judge端点与预算。不要把这些未知项当成已部署能力。
