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



# 适配 HotpotQA 格式
def clean_and_prepare_hotpotqa(hotpot_path, min_chunk_len=30, sentences_per_chunk=3, overlap_sentences=1):
    with open(hotpot_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in tqdm(data, desc="预处理HotpotQA样本"):
        question = item.get("question", "").strip()
        gold = item.get("answer", "").strip()
        # 拼接所有 context 段落文本
        context_list = item.get("context", [])
        context = " ".join([" ".join(paragraph[1]) for paragraph in context_list])
        # 切分chunks
        chunks = split_into_chunks(context, min_length=min_chunk_len, sentences_per_chunk=sentences_per_chunk, overlap_sentences=overlap_sentences)
        if not chunks or not gold:
            continue
        samples.append({
            "question": question,
            "gold": gold,
            "chunks": chunks
        })
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/hotpot_train_v1.1.json")
    parser.add_argument("--output_path", type=str, default="data/answer_label_hotpot.json")
    parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--target_dim", type=int, default=256)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("清洗并准备HotpotQA数据...")
    samples = clean_and_prepare_hotpotqa(args.data_path)
    print(f"有效样本数: {len(samples)}")

    print("加载句向量模型...")
    embedder = SentenceTransformerEmbedder(args.embedder, target_dim=args.target_dim)
    rag = PureRAG(embedder, llm_generator=None)  # 不生成答案，只做检索


    # 收集所有样本的 chunks，构建全局检索库
    all_chunks = []
    for sample in samples:
        all_chunks.extend(sample["chunks"])

    # 分批计算全体 chunks 的 embedding，并加进度条
    batch_size = 1024
    all_chunk_embs = []
    print(f"总chunks数: {len(all_chunks)}，开始计算embedding...")
    import numpy as np
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding chunks"):
        batch = all_chunks[i:i+batch_size]
        batch_embs = embedder.embed_batch(batch)
        # 如果 batch_embs 是 torch tensor，转为 numpy
        if isinstance(batch_embs, torch.Tensor):
            batch_embs = batch_embs.detach().cpu().numpy()
        all_chunk_embs.extend(batch_embs)
    # 合并为numpy数组再转为Tensor，确保rag.retrieve参数正确
    all_chunk_embs = np.vstack(all_chunk_embs)
    all_chunk_embs = torch.from_numpy(all_chunk_embs)

    answer_label_data = []
    for sample in tqdm(samples, desc="RAG检索与标注"):
        question = sample["question"]
        gold = sample["gold"]
        # PureRAG检索（在所有文章片段中检索）
        context, details = rag.retrieve(question, all_chunks, all_chunk_embs, topk=1)
        # 更严格：只允许top1 chunk完全包含gold answer才算可检索
        top_chunk = context
        if gold.strip() == top_chunk.strip() or gold in top_chunk:
            label = "可检索"
        else:
            label = "不可检索"
        answer_label_data.append({"text": top_chunk, "label": label, "question": question, "gold": gold})

    # 直接保存全部结果为 answer_label_hotpot.json
    output_path = args.output_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answer_label_data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {output_path} (总样本数: {len(answer_label_data)})")

if __name__ == "__main__":
    main()