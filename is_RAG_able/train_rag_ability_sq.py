import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/answer_label_dataset.json")
    parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="is_RAG_able/rag_ability_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"加载标注数据集 {args.data_path} ...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        labeled_data = json.load(f)
    sentences = [item["text"] for item in labeled_data]
    labels = [1 if item["label"] == "可检索" else 0 for item in labeled_data]
    print(f"样本数量: {len(sentences)}")
    print("加载句向量模型：", args.embedder)
    embedder = SentenceTransformer(args.embedder)
    print("计算嵌入向量（可能占用内存）...")
    emb = embedder.encode(sentences, convert_to_tensor=True, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    dataset = TensorDataset(emb, labels_t)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = emb.size(-1)
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
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg:.4f}")

    print("保存模型...")
    meta = {"embedder": args.embedder, "input_dim": input_dim, "hidden_dim": 256}
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    with open(args.save_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"保存完成: {args.save_path}")

if __name__ == "__main__":
    main()
