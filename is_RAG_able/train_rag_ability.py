import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

print("脚本已启动：is_RAG_able/train_rag_ability.py")

class RAGAbilityDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data  # List of sentences (strings)
        self.targets = targets  # List of labels (0/1 or float)
import argparse
import json
import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

print("脚本已启动：is_RAG_able/train_rag_ability.py")


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
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


def prepare_data(sentences: List[str], labels: List[int], embedder: SentenceTransformer, device: torch.device):
    # Compute sentence embeddings in batches
    emb = embedder.encode(sentences, convert_to_tensor=True, device=device)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    dataset = TensorDataset(emb, labels_t)
    return dataset


def train(model, dataloader, criterion, optimizer, device, epochs=5):
    model.to(device)
    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg:.4f}")


def save_model(model: nn.Module, path: str, meta: dict = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    if meta is not None:
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def load_and_infer(sentence: str, model_path: str, embedder_name: str, threshold: float = 0.5, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    embedder = SentenceTransformer(embedder_name)
    emb = embedder.encode([sentence], convert_to_tensor=True, device=device)
    # We need input dim to recreate classifier architecture; infer from meta if exists
    meta_path = model_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        input_dim = meta.get("input_dim")
        hidden_dim = meta.get("hidden_dim", 256)
    else:
        input_dim = emb.size(-1)
        hidden_dim = 256
    model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with torch.no_grad():
        logit = model(emb)
        prob = torch.sigmoid(logit).item()
    is_rag_able = prob >= threshold
    return {"prob": prob, "is_rag_able": is_rag_able}


def clean_squad_json(json_data):
    """
    读取SQuAD格式的json，返回所有question和label（有answer为1，无为0）
    """
    sentences = []
    labels = []
    for article in json_data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                q = qa.get("question", "")
                ans = qa.get("answers", [])
                label = 1 if ans and len(ans) > 0 else 0
                sentences.append(q)
                labels.append(label)
    return sentences, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="is_RAG_able/rag_ability_model.pt")
    parser.add_argument("--max_samples", type=int, default=20000, help="limit number of training samples for quick runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("加载本地数据集 data/train-v2.0.json ...")
    with open("data/train-v2.0.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
    sentences, labels = clean_squad_json(json_data)
    if args.max_samples and args.max_samples > 0:
        sentences = sentences[: args.max_samples]
        labels = labels[: args.max_samples]

    print(f"样本数量: {len(sentences)}")
    print("加载句向量模型：", args.embedder)
    embedder = SentenceTransformer(args.embedder)
    print("计算嵌入向量（可能占用内存）...")
    dataset = prepare_data(sentences, labels, embedder, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = dataset.tensors[0].size(-1)
    model = MLPClassifier(input_dim=input_dim)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("开始训练...")
    train(model, dataloader, criterion, optimizer, device, epochs=args.epochs)

    print("保存模型...")
    meta = {"embedder": args.embedder, "input_dim": input_dim, "hidden_dim": 256}
    save_model(model, args.save_path, meta=meta)
    print(f"保存完成: {args.save_path}")


if __name__ == "__main__":
    main()
