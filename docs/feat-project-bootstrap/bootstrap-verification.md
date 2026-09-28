# 项目初始化验收

- GitHub fork已核实：parent=XiaomiMiMo/verl；source=verl-project/verl；public。
- 源码起点a2ad9f6160b03ff2d47e59832bfb6b289f37c917；没有修改训练算法或环境逻辑。
- ruff check .：All checks passed。
- ruff format --check .：720 files already formatted。
- ruff format .：最小修复后720 files left unchanged。
- 9个Python格式化文件与HEAD比较AST完全一致。
- 2个notebook输出、metadata、execution_count、单元格数量和非代码内容完全一致；导入排序、unused requests、注释换行及局部noqa单独审阅。
- git diff --check通过。
- HTML来自已验收专项报告，新增建仓状态与Harbor/Pro边界说明；不依赖外部资源。
- GPU训练、Ray/K8s部署、模型导出和论文评测未执行。

推送前门禁再次运行；远端main和默认分支以最终GitHub核验为准。
