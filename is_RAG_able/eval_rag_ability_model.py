import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, accuracy_score
import os

def load_meta(model_path):
    meta_path = model_path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences = [item["text"] for item in data]
    labels = [1 if item["label"] == "可检索" else 0 for item in data]
    return sentences, labels

def eval_model(model_path, data_path, batch_size=64, device=None):
    device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    meta = load_meta(model_path)
    embedder_name = meta["embedder"] if meta and "embedder" in meta else "all-MiniLM-L6-v2"
    input_dim = meta["input_dim"] if meta and "input_dim" in meta else 384
    hidden_dim = meta["hidden_dim"] if meta and "hidden_dim" in meta else 256

    sentences, labels = load_data(data_path)
    embedder = SentenceTransformer(embedder_name)
    X_emb = embedder.encode(sentences, convert_to_tensor=True, device=device)
    y_t = torch.tensor(labels, dtype=torch.float32, device=device)
    dataset = TensorDataset(X_emb, y_t)
    loader = DataLoader(dataset, batch_size=batch_size)

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

    model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["不可检索", "可检索"])
    return acc, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="is_RAG_able/rag_ability_model.pt")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("评估训练集...")
    train_acc, train_report = eval_model(args.model_path, args.train_data, args.batch_size, device)
    print(f"Train Accuracy: {train_acc:.4f}")
    print("Train Classification Report:")
    print(train_report)
    print("评估测试集...")
    test_acc, test_report = eval_model(args.model_path, args.test_data, args.batch_size, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Test Classification Report:")
    print(test_report)
    with open("is_RAG_able/eval_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Train Accuracy: {train_acc:.4f}\n")
        f.write("Train Classification Report:\n")
        f.write(train_report + "\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write("Test Classification Report:\n")
        f.write(test_report)

if __name__ == "__main__":
    main()
