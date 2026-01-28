# Eval: Three Modes Report

**运行配置**

- **脚本**: projects/hybrid_titans/eval_three_modes.py
- **参数**: num_docs=20, doc_len=32, dim=64, digest_steps=200, seed=42

**结果摘要**

- **纯 Titan (Titan-only)**: fidelity: **98.20%**, loss: **0.0002155334**
- **Titan + RAG (Hybrid full)**: fidelity: **100.00%**, loss: **0.0**
- **Titan 与 RAG 门控融合 (Decision-gated)**: fidelity: **99.32%**, loss: **0.00008158295**
  - **gate**: 0.3847629850319665
  - **retr_score**: 0.3018115571149263
  - **mem_sim**: 0.3018115571148885
  - **surprise**: 0.5417605767607939

**指标说明**

- **Loss**: 使用 0.5 * MSE（均方误差）的平均值作为误差度量；数值越小表示输出与目标越接近。
- **Fidelity**: 一个归一化百分比指标，定义为 100 * (1 - (2 * loss) / (目标平均能量 + 1e-9))，值越接近 100 表示输出更忠实于目标向量。
- **gate**: 门控权重（0..1），表示在决策融合中对上下文输出 (`context_out`) 的权重；`1 - gate` 为对记忆输出 (`mem_out`) 的权重。
- **retr_score**: 检索得分，表示查询与文档向量的余弦相似度（越高表示检索上下文越相关）。
- **mem_sim**: 当前记忆预测与查询的相似度（余弦）。
- **surprise**: 记忆预测与查询之间的 MSE（用于衡量记忆的不确定或“惊讶”程度）。

**结果解读（简要）**

- **Hybrid full（Titan + RAG）表现最好**，在该模拟中为完美输出（fidelity 100%，loss 0），这是因为模拟中假定混合模型能“完美看到目标”，作为上界参考。\
- **Titan-only** 达到很高的 fidelity（98.20%），表明单纯通过 Titan 的消化过程也能较好地拟合目标表示，但仍略逊于混合上下文访问。\
- **Decision-gated 融合** 在 fidelity（99.32%）上接近混合全通道，且低于完全完美但优于纯 Titan；门控值 ~0.385 表明此样本中系统更多依赖记忆输出（约 61.5%）而不是上下文（约 38.5%）。造成门控偏向记忆的原因包括较高的 `surprise`（0.5418），在门控公式中 surprise 会降低 gate，从而倾向使用记忆以降低风险。

**建议的后续工作**

- 对多个随机种子或不同参数（`digest_steps`, `num_docs`, `dim`）运行多次并统计平均/方差，以评估稳定性和显著性。 
- 若需可视化对比，导出为 CSV 并绘制 fidelity 与 gate 的箱线图或散点图。
- 如果你希望我把该报告合入仓库 README 或生成 CSV，我可以继续执行。

---

报告文件位置: [projects/hybrid_titans/eval_three_modes_report.md](projects/hybrid_titans/eval_three_modes_report.md)
