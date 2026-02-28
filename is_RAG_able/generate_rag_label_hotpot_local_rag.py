import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/hybrid_titans/common')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../projects/hybrid_titans')))
from embedders import SentenceTransformerEmbedder
from text_utils import split_into_chunks
from baselines import PureRAG

def clean_and_prepare_hotpotqa(hotpot_path, min_chunk_len=30, sentences_per_chunk=3, overlap_sentences=1):
    with open(hotpot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in tqdm(data, desc="预处理HotpotQA样本"):
        question = item.get("question", "").strip()
        gold = item.get("answer", "").strip()
        context_list = item.get("context", [])
        context = " ".join([" ".join(paragraph[1]) for paragraph in context_list])
        chunks = split_into_chunks(context, min_length=min_chunk_len, sentences_per_chunk=sentences_per_chunk, overlap_sentences=overlap_sentences)
        if not chunks or not gold:
            continue
        samples.append({
            "question": question,
            "gold": gold,
            "chunks": chunks
        })
    return samples

def add_distractor_chunks(chunks, all_chunks, num_distractor=5):
    # 随机选取干扰chunk，避免和本身重复
    import random
    distractors = []
    chunk_set = set(chunks)
    candidates = [c for c in all_chunks if c not in chunk_set]
    if len(candidates) > num_distractor:
        distractors = random.sample(candidates, num_distractor)
    else:
        distractors = candidates
    return chunks + distractors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/hotpot_train_v1.1.json")
    parser.add_argument("--output_path", type=str, default="data/answer_label_hotpot_localrag.json")
    parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--target_dim", type=int, default=256)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--num_distractor", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("清洗并准备HotpotQA数据...")
    samples = clean_and_prepare_hotpotqa(args.data_path)
    print(f"有效样本数: {len(samples)}")

    print("加载句向量模型...")
    embedder = SentenceTransformerEmbedder(args.embedder, target_dim=args.target_dim)
    rag = PureRAG(embedder, llm_generator=None)

    # 收集所有chunks用于干扰项
    all_chunks = []
    for sample in samples:
        all_chunks.extend(sample["chunks"])

    answer_label_data = []
    for sample in tqdm(samples, desc="Local RAG检索与标注"):
        question = sample["question"]
        gold = sample["gold"]
        # 构建本地检索库（本样本chunks+干扰项）
        local_chunks = add_distractor_chunks(sample["chunks"], all_chunks, num_distractor=args.num_distractor)
        local_chunk_embs = embedder.embed_batch(local_chunks)
        # 只在本地检索库中检索
        context, details = rag.retrieve(question, local_chunks, local_chunk_embs, topk=1)
        top_chunk = context
        if gold.strip() == top_chunk.strip() or gold in top_chunk:
            label = "可检索"
        else:
            label = "不可检索"
        answer_label_data.append({"text": top_chunk, "label": label, "question": question, "gold": gold})

    output_path = args.output_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answer_label_data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {output_path} (总样本数: {len(answer_label_data)})")

if __name__ == "__main__":
    main()
