"""
对比 gate 回归决策器与原有手工公式门控的效果。
评测指标：平均 fidelity、loss，gate 分布等。
输出：eval_decision_maker_vs_formula_report.md
"""
import pandas as pd
import joblib
import numpy as np

# 加载数据和模型
csv_path = 'decision_maker_essays.csv'
df = pd.read_csv(csv_path)
reg = joblib.load('decision_maker_gate_reg.pkl')

# 原有公式门控
w1, w2, w3, b = 1.0, 0.8, 1.5, -0.2
def formula_gate(row):
    x = w1 * row['retr_score'] + w2 * row['mem_sim'] - w3 * row['surprise'] + b
    return 1.0 / (1.0 + np.exp(-x))

def clamp(x):
    return max(0.0, min(1.0, x))

# 评测
fids_reg, fids_formula = [], []
for _, row in df.iterrows():
    # 真实 gate 融合 fidelity 作为上界
    g_true = row['gate']
    # 用回归模型预测 gate
    g_reg = clamp(reg.predict([[row['retr_score'], row['mem_sim'], row['surprise']]])[0])
    # 用公式预测 gate
    g_formula = clamp(formula_gate(row))
    # 假设 context_out=1, mem_out=0, target=1，fidelity 近似为 gate 越接近真实 gate 越好
    # 这里用 gate 与真实 gate 的距离反映 fidelity
    fid_reg = 100.0 - 100.0 * abs(g_reg - g_true)
    fid_formula = 100.0 - 100.0 * abs(g_formula - g_true)
    fids_reg.append(fid_reg)
    fids_formula.append(fid_formula)

mean_reg = np.mean(fids_reg)
mean_formula = np.mean(fids_formula)

with open('eval_decision_maker_vs_formula_report.md', 'w', encoding='utf-8') as f:
    f.write(f"# 决策器 vs 手工公式门控效果对比报告\n\n")
    f.write(f"**平均 fidelity（回归决策器）**: {mean_reg:.2f}%\n\n")
    f.write(f"**平均 fidelity（手工公式）**: {mean_formula:.2f}%\n\n")
    f.write(f"\n**分布统计（前10条）**\n\n")
    f.write("| 题目 | 真实gate | 回归gate | 公式gate | 回归fidelity | 公式fidelity |\n")
    f.write("|------|----------|----------|----------|--------------|--------------|\n")
    for i, row in df.head(10).iterrows():
        g_true = row['gate']
        g_reg = clamp(reg.predict([[row['retr_score'], row['mem_sim'], row['surprise']]])[0])
        g_formula = clamp(formula_gate(row))
        fid_reg = 100.0 - 100.0 * abs(g_reg - g_true)
        fid_formula = 100.0 - 100.0 * abs(g_formula - g_true)
        f.write(f"| {row['question'][:12]}... | {g_true:.3f} | {g_reg:.3f} | {g_formula:.3f} | {fid_reg:.2f} | {fid_formula:.2f} |\n")
    f.write("\n\n**注：fidelity 这里用 gate 预测与真实 gate 的接近程度近似衡量，越高越好。**\n")
print('评测完成，报告已生成 eval_decision_maker_vs_formula_report.md')
