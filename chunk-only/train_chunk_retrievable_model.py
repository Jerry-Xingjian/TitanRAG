import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import ParameterGrid

# 1. 加载数据
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/chunk_feature_label_balanced.json')
parser.add_argument('--output_model', type=str, default='chunk-only/chunk_retrievable_rf.pkl')
parser.add_argument('--report_path', type=str, default='chunk-only/chunk_retrievable_report.txt')
args = parser.parse_args()


tqdm.write(f"加载数据集: {args.data_path}")
with open(args.data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
tqdm.write(f"总样本数: {len(data)}")

# 2. 提取特征和标签

features = []
labels = []
tqdm.write("提取特征...")
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



# 先转为numpy数组再打印
X = np.array(features)
y = np.array(labels)
tqdm.write(f"特征维度: {X.shape[1]}, 正负样本: {np.sum(y==1)}/{np.sum(y==0)}")


# 3. 划分训练集和测试集
tqdm.write("划分训练/测试集...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
tqdm.write(f"训练集: {X_train.shape[0]}，测试集: {X_test.shape[0]}")

# 4. 特征归一化
tqdm.write("归一化特征...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

tqdm.write("特征选择...")
selector_model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, tree_method='hist')
selector_model.fit(X_train, y_train)
selector = SelectFromModel(selector_model, prefit=True, threshold='median')
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)
selected_idx = selector.get_support(indices=True)
tqdm.write(f"选中特征数: {X_train_sel.shape[1]}")

# 5. 训练模型（调参提升泛化能力）
tqdm.write("训练随机森林模型...")

clf = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    # 提高对不可检索项(0)的惩罚，scale_pos_weight<1
    scale_pos_weight=0.5,  # 0.5表示对正类(1,可检索)损失减半，负类(0,不可检索)损失加倍
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=1
)
from time import sleep
tqdm.write("训练XGBoost模型/网格搜索调参...")
param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [6, 10, 14],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'scale_pos_weight': [0.5, 1.0]
}
base_clf = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss',
    verbosity=1
)
tqdm.write("[GridSearchCV] 这一步可能需要几分钟，请耐心等待...")
from sklearn.model_selection import cross_val_score
tqdm.write("[GridSearchCV] 这一步可能需要几分钟，请耐心等待...")
best_score = -1
best_params = None
best_clf = None
results = []
param_list = list(ParameterGrid(param_grid))
with tqdm(total=len(param_list), desc="GridSearchCV进度", mininterval=1) as pbar:
    for params in param_list:
        clf = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            tree_method='hist',
            eval_metric='logloss',
            verbosity=0,
            **params
        )
        scores = cross_val_score(clf, X_train_sel, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        mean_score = scores.mean()
        results.append((params, mean_score))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_clf = clf
        pbar.update(1)
tqdm.write(f"最佳参数: {best_params}")
tqdm.write(f"最佳交叉验证准确率: {best_score:.4f}")
best_clf.fit(X_train_sel, y_train, eval_set=[(X_test_sel, y_test)], verbose=True)
clf = best_clf

# 特征选择
tqdm.write("特征选择...")
selector_model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, tree_method='hist')
selector_model.fit(X_train, y_train)
selector = SelectFromModel(selector_model, prefit=True, threshold='median')
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)
selected_idx = selector.get_support(indices=True)
tqdm.write(f"选中特征数: {X_train_sel.shape[1]}")

# 网格搜索自动调参
tqdm.write("网格搜索调参...")
tqdm.write("[GridSearchCV] 这一步可能需要几分钟，请耐心等待...")
param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [6, 10, 14],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'scale_pos_weight': [0.5, 1.0]
}
base_clf = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss',
    verbosity=1
)
grid = GridSearchCV(base_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid.fit(X_train_sel, y_train)
clf = grid.best_estimator_
tqdm.write(f"最佳参数: {grid.best_params_}")
tqdm.write(f"最佳交叉验证准确率: {grid.best_score_:.4f}")
clf.fit(X_train_sel, y_train, eval_set=[(X_test_sel, y_test)], verbose=True)

tqdm.write("模型预测...")
y_pred = clf.predict(X_test_sel)

# 训练集准确率
y_train_pred = clf.predict(X_train_sel)
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)
tqdm.write(f'Train Accuracy: {train_acc:.4f}')
tqdm.write(f'Test Accuracy: {test_acc}')
tqdm.write(report)

# 特征重要性输出（只显示选中的特征）
importances = clf.feature_importances_
tqdm.write("特征重要性（选中特征）:")
for idx, val in zip(selected_idx, importances):
    tqdm.write(f"特征{idx}: {val:.4f}")

# 保存测试报告
with open(args.report_path, 'w', encoding='utf-8') as f:
    f.write(f'Train Accuracy: {train_acc:.4f}\n')
    f.write(f'Test Accuracy: {test_acc:.4f}\n')
    f.write(report)
    f.write("\n最佳参数:\n")
    f.write(str(grid.best_params_)+"\n")
    f.write(f"最佳交叉验证准确率: {grid.best_score_:.4f}\n")
    f.write("\n特征重要性（选中特征）:\n")
    for idx, val in zip(selected_idx, importances):
        f.write(f"特征{idx}: {val:.4f}\n")
tqdm.write(f'Report saved to {args.report_path}')

import joblib
joblib.dump(clf, args.output_model)
tqdm.write(f'Model saved to {args.output_model}')
