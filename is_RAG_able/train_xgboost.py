import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.spatial.distance import euclidean, cityblock
import xgboost as xgb

def compute_bm25_scores(questions, texts):
    vectorizer = TfidfVectorizer().fit(texts + questions)
    text_tfidf = vectorizer.transform(texts)
    q_tfidf = vectorizer.transform(questions)
    scores = (q_tfidf * text_tfidf.T).toarray()
    return np.diag(scores), scores, q_tfidf, text_tfidf
   
def compute_cosine_sim(q_embs, s_embs):
    q_norm = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-8)
    s_norm = s_embs / (np.linalg.norm(s_embs, axis=1, keepdims=True) + 1e-8)
    return np.sum(q_norm * s_norm, axis=1, keepdims=True)

def compute_euclidean(q_embs, s_embs):
    return np.linalg.norm(q_embs - s_embs, axis=1, keepdims=True)

def compute_manhattan(q_embs, s_embs):
    return np.sum(np.abs(q_embs - s_embs), axis=1, keepdims=True)

def compute_jaccard(qs, ss):
    def jaccard(a, b):
        set_a = set(a.split())
        set_b = set(b.split())
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)
    return np.array([jaccard(q, s) for q, s in zip(qs, ss)]).reshape(-1, 1)

def compute_tfidf_cosine(q_tfidf, text_tfidf):
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(q_tfidf, text_tfidf)
    return np.diag(sims).reshape(-1, 1)

def extract_question_type(qs):
    types = []
    for q in qs:
        q = q.strip()
        if q.startswith("什么"): types.append("what")
        elif q.startswith("谁"): types.append("who")
        elif q.startswith("多少") or q.startswith("几"): types.append("how_many")
        elif q.startswith("哪"): types.append("which")
        elif q.startswith("为何") or q.startswith("为什么"): types.append("why")
        elif q.startswith("何时") or q.startswith("什么时候"): types.append("when")
        else: types.append("other")
    lb = LabelBinarizer()
    return lb.fit_transform(types)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/answer_label_sq.json")
    parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        labeled_data = json.load(f)
    sentences = [item["text"] for item in labeled_data]
    labels = [1 if item["label"] == "可检索" else 0 for item in labeled_data]
    questions = [item.get("question", "") for item in labeled_data]
    golds = [item.get("gold", "") for item in labeled_data]
    print(f"Samples: {len(sentences)}")
    print("Loading embedder:", args.embedder)
    embedder = SentenceTransformer(args.embedder)
    print("Encoding embeddings...")
    emb = embedder.encode(sentences, convert_to_tensor=True)
    q_emb = embedder.encode(questions, convert_to_tensor=True)
    q_lens = np.array([len(q) for q in questions]).reshape(-1, 1)
    gold_lens = np.array([len(g) for g in golds]).reshape(-1, 1)
    # 分批计算BM25和TF-IDF相关特征，避免内存溢出
    batch_size = 2048
    bm25_diag_list, bm25_rank_list, cos_sim_list, eu_dist_list, man_dist_list, jaccard_sim_list, tfidf_cos_list = [], [], [], [], [], [], []
    emb_np = emb.cpu().numpy() if hasattr(emb, 'cpu') else emb
    q_emb_np = q_emb.cpu().numpy() if hasattr(q_emb, 'cpu') else q_emb
    for i in range(0, len(questions), batch_size):
        q_batch = questions[i:i+batch_size]
        s_batch = sentences[i:i+batch_size]
        # BM25/TF-IDF
        bm25_diag, bm25_matrix, q_tfidf, text_tfidf = compute_bm25_scores(q_batch, s_batch)
        bm25_diag_list.append(bm25_diag.reshape(-1, 1))
        bm25_rank_list.append(np.argsort(-bm25_matrix, axis=1)[:, 0].reshape(-1, 1))
        # 其他特征
        cos_sim_list.append(compute_cosine_sim(q_emb_np[i:i+batch_size], emb_np[i:i+batch_size]))
        eu_dist_list.append(compute_euclidean(q_emb_np[i:i+batch_size], emb_np[i:i+batch_size]))
        man_dist_list.append(compute_manhattan(q_emb_np[i:i+batch_size], emb_np[i:i+batch_size]))
        jaccard_sim_list.append(compute_jaccard(q_batch, s_batch))
        tfidf_cos_list.append(compute_tfidf_cosine(q_tfidf, text_tfidf))
    bm25_diag = np.vstack(bm25_diag_list)
    bm25_rank = np.vstack(bm25_rank_list)
    cos_sim = np.vstack(cos_sim_list)
    eu_dist = np.vstack(eu_dist_list)
    man_dist = np.vstack(man_dist_list)
    jaccard_sim = np.vstack(jaccard_sim_list)
    tfidf_cos = np.vstack(tfidf_cos_list)
    q_type_oh = extract_question_type(questions)
    num_feats = np.concatenate([
        q_lens, gold_lens, bm25_diag, bm25_rank, cos_sim, eu_dist, man_dist, jaccard_sim, tfidf_cos
    ], axis=1)
    scaler = StandardScaler()
    num_feats = scaler.fit_transform(num_feats)
    features = np.concatenate([emb_np, num_feats, q_type_oh], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=args.test_size, random_state=args.seed, stratify=labels)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    # 自动调优参数
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    base_clf = xgb.XGBClassifier(random_state=args.seed, n_jobs=-1, tree_method='hist')
    grid = GridSearchCV(base_clf, param_grid, cv=3, scoring='accuracy', verbose=2)
    grid.fit(X_train, y_train)
    clf = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_report = classification_report(y_train, y_pred_train, target_names=["NotRetrievable", "Retrievable"])
    test_report = classification_report(y_test, y_pred_test, target_names=["NotRetrievable", "Retrievable"])
    print(f"Train Accuracy: {train_acc:.4f}")
    print(train_report)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(test_report)
    # 输出特征重要性
    importances = clf.feature_importances_
    print("Feature importances:", importances)
    with open("is_RAG_able/xgb_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write("Train Classification Report:\n")
        f.write(train_report + "\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write("Test Classification Report:\n")
        f.write(test_report)
        f.write("\n# Metric explanations:\n")
        f.write("# precision: Of all samples predicted as this class, the proportion that are actually this class (true positive rate among predicted positives).\n")
        f.write("# recall: Of all actual samples of this class, the proportion that are correctly predicted (true positive rate among actual positives).\n")
        f.write("# f1-score: Harmonic mean of precision and recall, reflecting the balance between them.\n")
        f.write("# support: The number of true samples of this class in the dataset.\n")
        f.write("# accuracy: Overall proportion of correctly predicted samples.\n")
        f.write("# macro avg: Unweighted mean of the metric across all classes.\n")
        f.write("# weighted avg: Mean of the metric across all classes, weighted by the number of true samples for each class.\n")
        f.write("\nBest params:\n")
        f.write(str(grid.best_params_) + "\n")
        f.write("Best CV score: " + str(grid.best_score_) + "\n")
        f.write("Feature importances:\n")
        f.write(str(importances) + "\n")

        # 保存模型
        import joblib
        joblib.dump(clf, "is_RAG_able/xgb_model.pkl")

if __name__ == "__main__":
    main()
