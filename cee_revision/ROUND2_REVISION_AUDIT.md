# 第二轮一区审稿意见修改审计

## 结论

本轮审稿意见已逐项落实到独立版本。修改没有把回顾性分析包装成前瞻性验证，也没有通过收缩结论回避不确定性；新增证据直接检验支持门控、效用离散度、部署层级与指标口径。

## 主要意见落实

1. **post hoc / retrospective / prospective 术语边界**：摘要、方法、结果、讨论、图文摘要和补充材料均明确说明，支持门控是在冻结实验完成后提出并回顾性审计，只能作为未来前瞻性流程的候选规则。
2. **支持门控阈值敏感性**：完成 3×3×3 共 27 个配置，交叉改变最小事件数 15/25/40、重复 AUROC 中位数 0.60/0.65/0.70、阈值最大范围 0.05/0.10/0.15。通过数为 1–7/10，预防率 95.2%–97.9%，保留率 1.4%–8.6%，等成本效用 +0.35 至 +13.75/10,000，最差批次效用 −3.17 至 +0.43。
3. **部署层级**：只有 conditional Lite-CF 接受回顾性支持门控；不通过时直接回退到 PDRF，不能自动切换到 all-row 或 two-stage。后两者保持为分析控制，2:1 的 two-stage 结果不再表述为部署建议。
4. **拟合对离散度**：对 10 个拟合对报告均值、Student-t 95% 区间、中位数、四分位距、正效用计数、选择率、负迁移预防率、纠正保留率、校准效用及 test-minus-calibration 变化。
5. **0.720 与 0.562 的口径**：前者是 100 个 seed-repeat AUROC 的算术平均，后者是把正式 OOF 行拼接后计算的 pooled-row AUROC；正文和补充材料已明确区分。
6. **Figure 3a 的性质**：改为 descriptive cell-stratified test ranking curves，并明确这些曲线不是可部署策略，只有校准冻结阈值对应可提交的决策点。
7. **类别代价限制**：讨论中明确当前效用把纠正和伤害视为类别无关的二元事件，没有建模 class-to-class 代价矩阵，也没有区分两种错误端点的代价差异。
8. **方法与核心基线不平衡**：新增精简端点基准表及架构选择理由；主效用表补充 two-stage 1:1，并统一“utility units”口径。

## 独立数值核验

- 正式预测文件：57,440 行，无空值、无重复主键；10 个拟合对×4 个固定故障机制，每个单元 1,436 行。
- 支持门控网格：27 个唯一配置；参考配置 25/0.65/0.10 通过 5/10 个拟合对，负迁移预防率 96.2%，纠正保留率 6.6%，效用 +7.49/10,000。
- 2:1 two-stage：测试效用均值 +37.78，拟合对 95% 区间 −2.71 至 +78.27，7/10 为正；校准均值 +136.77，test-minus-calibration 为 −98.99。
- 汇总表共 32 个 method×cost 条件，每项均有 10 个拟合对；未发现 NaN、Inf 或重复配置。

## 版面与文件 QA

- 主文 PDF：33 页，无 overfull box、未定义引用或未定义文献。
- 补充材料 PDF：50 页，无 overfull box、未定义引用或未定义文献。
- 图文摘要：PNG/PDF/SVG/600-dpi TIFF 均已更新；术语改为 post hoc retrospective audit。
- 可编辑 Word：主文关键 10 列效用表已重建为固定列宽的原生 Word 表格，不再逐字断列或重叠；主文和补充材料均完成 Word-to-PDF 渲染检查。
- 未加入生成式 AI 声明；作者仍为 Riyang Luo 一人，经费和利益冲突均为无。

## 机器可读新增证据

- `analyse_cee_round2_revision.py`
- `source_data/round2_gate_sensitivity_full.csv`
- `source_data/round2_gate_sensitivity_onefactor.csv`
- `source_data/round2_utility_fitted_pair_metrics.csv`
- `source_data/round2_utility_fitted_pair_summary.csv`
- `tables/round2_*.tex`

