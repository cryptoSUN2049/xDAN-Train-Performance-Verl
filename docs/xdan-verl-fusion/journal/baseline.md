# Baseline 线观察日志（只读）

baseline 线由 xdan-train-performance-verl-13 会话负责，目标见它的 `AUTORUN-GOALS-20261002.md`。本文件只记录融合线看门狗的只读观察。分级规则见 [README.md](README.md)。时间一律为 UTC。

## 分析记录（L3）

### 2026-10-03 07:44 分析：General 4h 评测分数无效

- val-core 均值为几百的负数，因为 judge 崩溃，失败样本记为 −999，又被平均进去。baseline 会话已确认根因：RL 评测中 116 个样本有 62 个无效，作废；SFT 剔除失败样本后为 0.174。约 16:00 起在新目录重测（`*-r2`）。
- 截至 10:03 的有效结果：Webdev +7.3pt（0.536 → 0.610），Music −2.4pt（噪声范围内），Code SFT 0.422。

---

## 运行日志（L2，看门狗自动追加）
