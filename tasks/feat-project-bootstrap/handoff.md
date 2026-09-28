# feat-project-bootstrap / 冷启动交接

## 1. TL;DR
- xDAN-Train-Performance-Verl已建立为XiaomiMiMo/verl的fork。
- main为xDAN集成分支，mimo-oss保留原始基线；mimo和verl-upstream已配置。
- 专项HTML、MiMo/Ornith证据、复现教程、P0任务和资产manifest已归档。
- 仅最小lint/格式修复；未执行GPU训练。下一步是选定硬件/领域并开展P0。

## 2. 本轮交付物
- `README.md` — 356 行；项目入口/研究/复现与版本记录。
- `.gitignore` — 140 行；项目入口/研究/复现与版本记录。
- `docs/feat-project-bootstrap/index.html` — 2050 行；项目入口/研究/复现与版本记录。
- `docs/feat-project-bootstrap/design.md` — 38 行；项目入口/研究/复现与版本记录。
- `docs/feat-project-bootstrap/report-design.md` — 126 行；项目入口/研究/复现与版本记录。
- `docs/feat-project-bootstrap/mimo-research.md` — 176 行；项目入口/研究/复现与版本记录。
- `docs/feat-project-bootstrap/ornith-research.md` — 126 行；项目入口/研究/复现与版本记录。
- `docs/feat-project-bootstrap/verification.md` — 22 行；项目入口/研究/复现与版本记录。
- `docs/feat-project-bootstrap/bootstrap-verification.md` — 14 行；项目入口/研究/复现与版本记录。
- `docs/feat-project-bootstrap/p0-plan.md` — 29 行；项目入口/研究/复现与版本记录。
- `docs/feat-project-bootstrap/upstream-manifest.json` — 31 行；项目入口/研究/复现与版本记录。
- `tasks/todo.md` — 13 行；项目入口/研究/复现与版本记录。
- 9个Python文件只格式化，2个tutorial notebook最小lint修复；准确列表见独立style提交。
- `tasks/feat-project-bootstrap/handoff.md` — 本交接文件。

## 3. 设计约束
- 原始LICENSE/Notice和上游README内容保留；代码来源关系不可丢。
- 训练算法与适配层解耦，初期不改verl核心；上游升级先在隔离分支验证。
- 文档集中docs/当前worktree名，流程交接集中tasks/当前worktree名。
- 不将Pro训练体系、Ornith方法介绍、多个领域checkpoint混为公开9B统一方案。
- 每次push前ruff check .与ruff format --check .均须通过；不关闭规则绕过。

## 4. 真实行为与已踩坑
- 原始MiMo提交有8项notebook lint、9个Python格式问题；已最小修复。
- ruff --fix会重序列化notebook导致大diff；本轮恢复原JSON格式，只改code cell source。
- Code wrapper覆盖基础YAML；General公开数据树与脚本默认bundle不一致；image mapping不是拼prefix。
- Code preflight连Ray并probe GPU；General preflight不等同环境/judge验收。
- 公开9B只有SFT权重被找到，各域RL成品/内部eval资产仍有缺口。
- Harbor相关注释不等于Harbor主训练pipeline；现有MiMo环境接口已存在。
- 本机无已验证训练容器、GPU/Ray/K8s profile；不得直接声称可以开训。

## 5. 下一里程碑
- [ ] 确认GPU预算、共享存储、Ray/K8s可用性与优先领域。
- [ ] 执行P0.1–P0.8（见p0-plan.md），先单任务环境和奖励，再短程GRPO。
- [ ] 若需Harbor，设计并验收单任务adapter，不同时重写优化算法。
- [ ] 待基线成立后再建设性能看板、调度或Ornith式动态任务生成。

## 6. 分支/部署状态
- origin：https://github.com/cryptoSUN2049/xDAN-Train-Performance-Verl。
- 默认集成分支main；上游基线mimo-oss；初始化工作分支worktree-feat-project-bootstrap保留。
- 原始base a2ad9f6160b03ff2d47e59832bfb6b289f37c917。
- 最终提交以git log为准；无训练部署；GPU测试未执行；本地静态门禁通过。
- 使用保留worktree方式退出；主目录停在main，后续修改继续另建worktree。

## 7. 冷启动 checklist
1. 先读本handoff、p0-plan.md、upstream-manifest.json。
2. git status / git log -3 / git worktree list / git remote -v。
3. 打开docs/feat-project-bootstrap/index.html，重点教程与§12长期建设。
4. 核对mimo-research.md中的依赖/数据/镜像缺口，不凭假设启动训练。
5. 在新worktree开始下一批修改，固定数据/模型/容器版本并记录验收。
