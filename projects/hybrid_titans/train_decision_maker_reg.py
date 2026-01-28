"""
用 sample_essays 采集的特征训练 gate 回归模型，保存为 decision_maker_gate_reg.pkl。
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('decision_maker_essays.csv')
X = df[['retr_score', 'mem_sim', 'surprise']].values
y = df['gate'].values

reg = LinearRegression()
reg.fit(X, y)

joblib.dump(reg, 'decision_maker_gate_reg.pkl')
print('训练完成，模型已保存为 decision_maker_gate_reg.pkl')
