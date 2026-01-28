"""
用 scikit-learn 训练逻辑回归决策器，输入 retr_score, mem_sim, surprise，输出 label。
模型保存为 decision_maker_lr.pkl。
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('decision_maker_train.csv')
X = df[['retr_score', 'mem_sim', 'surprise']].values
y = df['label'].values

clf = LogisticRegression()
clf.fit(X, y)

joblib.dump(clf, 'decision_maker_lr.pkl')
print('训练完成，模型已保存为 decision_maker_lr.pkl')
