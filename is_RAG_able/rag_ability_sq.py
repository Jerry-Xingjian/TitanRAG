import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/hybrid_titans/common')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/hybrid_titans')))
from embedders import SentenceTransformerEmbedder
from text_utils import split_into_chunks
from baselines import PureRAG

def clean_and_prepare_squad(squad_path, min_chunk_len=30, sentences_per_chunk=3, overlap_sentences=1):
    with open(squad_path, "r", encoding="utf-8") as f:
        squad = json.load(f)
    samples = []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            # 切分chunks
            chunks = split_into_chunks(context, min_length=min_chunk_len, sentences_per_chunk=sentences_per_chunk, overlap_sentences=overlap_sentences)
            if not chunks:
                continue
            for qa in para["qas"]:
                # 只保留有答案的样本
                answers = qa.get("answers", [])
                if not answers or not answers[0].get("text", "").strip():
                    continue
                question = qa["question"].strip()
                gold = answers[0]["text"].strip()
                samples.append({
                    "question": question,
                    "gold": gold,
                    "chunks": chunks
                })
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train-v2.0.json")
    parser.add_argument("--output_path", type=str, default="data/answer_label_dataset.json")
    parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--target_dim", type=int, default=256)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("清洗并准备SQuAD数据...")
    samples = clean_and_prepare_squad(args.data_path)
    print(f"有效样本数: {len(samples)}")

    print("加载句向量模型...")
    embedder = SentenceTransformerEmbedder(args.embedder, target_dim=args.target_dim)
    rag = PureRAG(embedder, llm_generator=None)  # 不生成答案，只做检索

    answer_label_data = []
    for sample in tqdm(samples, desc="RAG检索与标注"):
        question = sample["question"]
        gold = sample["gold"]
        chunks = sample["chunks"]
        # 计算chunks embedding
        chunk_embs = embedder.embed_batch(chunks)
        # PureRAG检索
        context, details = rag.retrieve(question, chunks, chunk_embs, topk=1)
        # 更严格：只允许top1 chunk完全包含gold answer才算可检索
        top_chunk = context
        if gold.strip() == top_chunk.strip() or gold in top_chunk:
            label = "可检索"
        else:
            label = "不可检索"
        answer_label_data.append({"text": top_chunk, "label": label, "question": question, "gold": gold})

    # 欠采样可检索样本，使两类数量平衡
    import random
    can = [item for item in answer_label_data if item["label"] == "可检索"]
    cannot = [item for item in answer_label_data if item["label"] == "不可检索"]
    if len(can) > len(cannot):
        random.seed(42)
        can = random.sample(can, len(cannot))
    balanced_data = can + cannot
    random.shuffle(balanced_data)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {args.output_path} (平衡后样本数: {len(balanced_data)})")

if __name__ == "__main__":
    main()