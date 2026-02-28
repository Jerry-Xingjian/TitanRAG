import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb
import joblib

# 更快的训练脚本，使用LightGBM和早停，去除冗余网格搜索和多次特征选择
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/chunk_feature_label_balanced.json')
parser.add_argument('--output_model', type=str, default='chunk-only/chunk_retrievable_lgbm.pkl')
parser.add_argument('--report_path', type=str, default='chunk-only/chunk_retrievable_lgbm_report.txt')
args = parser.parse_args()

tqdm.write(f"加载数据集: {args.data_path}")
with open(args.data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
tqdm.write(f"总样本数: {len(data)}")

features = []
labels = []
for item in tqdm(data, desc="特征提取", mininterval=1):
    feat = []
    feat.extend(item['embedding'])
    feat.append(item['length'])
    feat.append(item['keyword_density'])
    feat.append(item['unique_word_count'])
    feat.append(item['avg_word_length'])
    feat.append(item['stopword_ratio'])
    feat.append(item['punctuation_count'])
    feat.append(item['number_count'])
    feat.append(item['year_count'])
    feat.append(item['sentence_count'])
    feat.append(item['chunk_question_similarity'])
    features.append(feat)
    labels.append(1 if item['is_retrievable'] else 0)

X = np.array(features)
y = np.array(labels)
tqdm.write(f"特征维度: {X.shape[1]}, 正负样本: {np.sum(y==1)}/{np.sum(y==0)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
tqdm.write(f"训练集: {X_train.shape[0]}，测试集: {X_test.shape[0]}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# LightGBM 训练，早停，快速
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 32,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    # 让负类（不可检索，label=0）权重更大，假阳性惩罚更重
    'scale_pos_weight': 0.8  # <1，正类权重更小，负类权重更大
}
tqdm.write("训练LightGBM模型（含早停）...")
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_eval],
    valid_names=['train', 'eval'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(period=20)
    ]
)

# 预测与评估
y_pred = (gbm.predict(X_test, num_iteration=gbm.best_iteration) > 0.5).astype(int)
train_acc = accuracy_score(y_train, (gbm.predict(X_train, num_iteration=gbm.best_iteration) > 0.5).astype(int))
test_acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
tqdm.write(f'Train Accuracy: {train_acc:.4f}')
tqdm.write(f'Test Accuracy: {test_acc:.4f}')
tqdm.write(report)

# 保存报告
with open(args.report_path, 'w', encoding='utf-8') as f:
    f.write(f'Train Accuracy: {train_acc:.4f}\n')
    f.write(f'Test Accuracy: {test_acc:.4f}\n')
    f.write(report)
tqdm.write(f'Report saved to {args.report_path}')

# 保存模型
joblib.dump(gbm, args.output_model)
tqdm.write(f'Model saved to {args.output_model}')
