import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from scipy.spatial.distance import euclidean, cityblock
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/answer_label_dataset_balanced.json")
    parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_path", type=str, default="is_RAG_able/train_test_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"加载标注数据集 {args.data_path} ...")
    # 特征增强：拼接句向量+问题长度+gold长度+BM25分数+embedding余弦相似度
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
        # q_tfidf, text_tfidf: sparse matrices
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(q_tfidf, text_tfidf)
        return np.diag(sims).reshape(-1, 1)

    with open(args.data_path, "r", encoding="utf-8") as f:
        labeled_data = json.load(f)
    sentences = [item["text"] for item in labeled_data]
    labels = [1 if item["label"] == "可检索" else 0 for item in labeled_data]
    questions = [item.get("question", "") for item in labeled_data]
    golds = [item.get("gold", "") for item in labeled_data]
    print(f"样本数量: {len(sentences)}")
    print("加载句向量模型：", args.embedder)
    embedder = SentenceTransformer(args.embedder)
    print("计算嵌入向量...")
    emb = embedder.encode(sentences, convert_to_tensor=True, device=device)
    q_emb = embedder.encode(questions, convert_to_tensor=True, device=device)
    # 特征增强
    q_lens = np.array([len(q) for q in questions]).reshape(-1, 1)
    gold_lens = np.array([len(g) for g in golds]).reshape(-1, 1)
    bm25_diag, bm25_matrix, q_tfidf, text_tfidf = compute_bm25_scores(questions, sentences)
    bm25_diag = bm25_diag.reshape(-1, 1)
    bm25_rank = np.argsort(-bm25_matrix, axis=1)[:, 0].reshape(-1, 1)
    emb_np = emb.cpu().numpy() if hasattr(emb, 'cpu') else emb
    q_emb_np = q_emb.cpu().numpy() if hasattr(q_emb, 'cpu') else q_emb
    cos_sim = compute_cosine_sim(q_emb_np, emb_np)
    eu_dist = compute_euclidean(q_emb_np, emb_np)
    man_dist = compute_manhattan(q_emb_np, emb_np)
    jaccard_sim = compute_jaccard(questions, sentences)
    tfidf_cos = compute_tfidf_cosine(q_tfidf, text_tfidf)
    # 问题类型one-hot
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
    q_type_oh = extract_question_type(questions)
    # 标准化数值特征
    num_feats = np.concatenate([
        q_lens, gold_lens, bm25_diag, bm25_rank, cos_sim, eu_dist, man_dist, jaccard_sim, tfidf_cos
    ], axis=1)
    scaler = StandardScaler()
    num_feats = scaler.fit_transform(num_feats)
    features = np.concatenate([emb_np, num_feats, q_type_oh], axis=1)
    features_t = torch.tensor(features, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features_t, labels_t, test_size=0.2, random_state=42, stratify=labels)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    input_dim = features.shape[1]
    # 更深的MLP+BatchNorm+GELU+残差
    class MLPClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim=2048, dropout=0.2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
            self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
            self.bn4 = nn.BatchNorm1d(hidden_dim // 4)
            self.fc5 = nn.Linear(hidden_dim // 4, 1)
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            x1 = F.gelu(self.bn1(self.fc1(x)))
            x2 = F.gelu(self.bn2(self.fc2(x1))) + x1  # 残差
            x3 = F.gelu(self.bn3(self.fc3(x2)))
            x4 = F.gelu(self.bn4(self.fc4(x3)))
            x4 = self.dropout(x4)
            out = self.fc5(x4)
            return out.squeeze(-1)
    model = MLPClassifier(input_dim=input_dim)
    model.to(device)
    class_weight = torch.tensor([1.0, 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # 早停
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    print("开始训练...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg:.4f}")
        # 早停监控
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_avg = val_loss / len(test_loader.dataset)
        print(f"  Validation Loss: {val_avg:.4f}")
        if val_avg < best_loss:
            best_loss = val_avg
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    # 训练结束后加载最佳模型
    model.load_state_dict(torch.load(args.save_path))

    print("保存模型...")
    meta = {"embedder": args.embedder, "input_dim": input_dim, "hidden_dim": 256}
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    with open(args.save_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"保存完成: {args.save_path}")

    # 训练集评估
    model.eval()
    all_train_preds = []
    all_train_labels = []
    with torch.no_grad():
        for xb, yb in train_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu().numpy()
            all_train_preds.extend(preds.tolist())
            all_train_labels.extend(yb.cpu().numpy().tolist())
    train_acc = accuracy_score(all_train_labels, all_train_preds)
    train_report = classification_report(all_train_labels, all_train_preds, target_names=["不可检索", "可检索"])

    # 测试集评估
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["不可检索", "可检索"])
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print("Train Classification Report:")
    print(train_report)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("Test Classification Report:")
    print(report)
    # 保存到文件
    with open("is_RAG_able/test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write("Train Classification Report:\n")
        f.write(train_report + "\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write("Test Classification Report:\n")
        f.write(report)

if __name__ == "__main__":
    main()
