import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/answer_label_dataset.json")
    parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="is_RAG_able/train_test_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"加载标注数据集 {args.data_path} ...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        labeled_data = json.load(f)
    sentences = [item["text"] for item in labeled_data]
    labels = [1 if item["label"] == "可检索" else 0 for item in labeled_data]
    print(f"样本数量: {len(sentences)}")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"训练集: {len(X_train)}，测试集: {len(X_test)}")

    print("加载句向量模型：", args.embedder)
    embedder = SentenceTransformer(args.embedder)
    print("计算训练集嵌入向量...")
    X_train_emb = embedder.encode(X_train, convert_to_tensor=True, device=device)
    print("计算测试集嵌入向量...")
    X_test_emb = embedder.encode(X_test, convert_to_tensor=True, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(X_train_emb, y_train_t)
    test_dataset = TensorDataset(X_test_emb, y_test_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    input_dim = X_train_emb.size(-1)
    class MLPClassifier(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = MLPClassifier(input_dim=input_dim)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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

    print("保存模型...")
    meta = {"embedder": args.embedder, "input_dim": input_dim, "hidden_dim": 256}
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    with open(args.save_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"保存完成: {args.save_path}")

    # 测试集评估
    model.eval()
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
    print(f"\nTest Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
