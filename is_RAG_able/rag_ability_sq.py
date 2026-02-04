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

    def main():
        import os
        import sys
        import argparse
        import json
        import re
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
        try:
            from main import TitanRAG  # 假设TitanRAG可直接import
        except ImportError:
            TitanRAG = None

        parser = argparse.ArgumentParser()
        parser.add_argument("--rag_model_path", type=str, default="is_RAG_able/rag_ability_model.pt", help="TitanRAG模型路径")
        parser.add_argument("--data_path", type=str, default="data/train-v2.0.json", help="SQuAD格式数据集路径")
        parser.add_argument("--output_path", type=str, default="data/answer_label_dataset.json", help="输出答案句-标签文件")
        args = parser.parse_args()

        device = None
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
        except ImportError:
            print("torch not installed, device not set")

        # 加载TitanRAG模型（此处需根据实际初始化方式调整）
        # titan_model = ... # 加载底层模型
        # rag = TitanRAG(titan_model)
        rag = None  # TODO: 替换为实际TitanRAG实例

        # 加载SQuAD格式数据
        with open(args.data_path, "r", encoding="utf-8") as f:
            squad_data = json.load(f)

        answer_label_data = []
        for article in squad_data.get("data", []):
            for para in article.get("paragraphs", []):
                context = para.get("context", "")
                for qa in para.get("qas", []):
                    question = qa.get("question", "")
                    # 1. 先用RAG推理
                    rag_answer = None
                    if rag is not None:
                        # 伪代码：rag_answer = rag.query_with_context(question, ...)
                        pass
                    # 判断RAG是否能答（此处用rag_answer是否为空判断，实际需根据你的RAG输出逻辑调整）
                    if rag_answer:
                        answer_label_data.append({"text": rag_answer, "label": "可检索"})
                    else:
                        # 2. 检索原文段落，找标准答案
                        answers = qa.get("answers", [])
                        if answers:
                            gold = answers[0].get("text", "")
                            sents = re.split(r'[。！？!?.]', context)
                            found = None
                            for sent in sents:
                                if gold in sent:
                                    found = sent.strip()
                                    break
                            if found:
                                answer_label_data.append({"text": found, "label": "不可检索"})
                            else:
                                answer_label_data.append({"text": gold, "label": "不可检索"})
                        else:
                            continue

        # 保存结果
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(answer_label_data, f, ensure_ascii=False, indent=2)
        print(f"已保存: {args.output_path}")
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

    print("加载标注数据集 data/labeled_sentences.json ...")
    with open("data/labeled_sentences.json", "r", encoding="utf-8") as f:
        labeled_data = json.load(f)
    sentences = [item["text"] for item in labeled_data]
    labels = [item["rag_label"] for item in labeled_data]
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
