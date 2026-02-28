import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
# coding: utf-8
from collections import Counter
import re
import string
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

# 路径设置，确保可以import到项目内模块
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
            chunks = split_into_chunks(context, min_length=min_chunk_len, sentences_per_chunk=sentences_per_chunk, overlap_sentences=overlap_sentences)
            if not chunks:
                continue
            for qa in para["qas"]:
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

def keyword_density(text):
    words = text.split()
    counter = Counter(words)
    return len(counter) / len(words) if words else 0

# 新增特征函数
def unique_word_count(text):
    words = set(text.split())
    return len(words)

def avg_word_length(text):
    words = text.split()
    if not words:
        return 0
    return sum(len(w) for w in words) / len(words)

def stopword_ratio(text):
    stopwords = set([
        'a','an','the','is','are','was','were','be','been','being','have','has','had','do','does','did','will','would','could','should','may','might','must','shall','can','need','dare','to','of','in','for','on','with','at','by','from','as','into','through','during','before','after','above','below','between','under','again','further','then','once','and','or','but','if','because','until','while','that','which','who','whom','this','these','those','it','its','their','they'
    ])
    words = text.split()
    if not words:
        return 0
    stop_count = sum(1 for w in words if w.lower() in stopwords)
    return stop_count / len(words)

def punctuation_count(text):
    return sum(1 for c in text if c in string.punctuation)

def number_year_count(text):
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    return len(numbers), len(years)

def sentence_count(text):
    try:
        return len(sent_tokenize(text))
    except Exception:
        return text.count('.') + 1 if text else 0

def embedding_similarity(emb1, emb2):
    # emb1, emb2: 1D torch tensors or numpy arrays
    import numpy as np
    # 转为1D torch tensor
    if not isinstance(emb1, torch.Tensor):
        emb1 = torch.tensor(np.array(emb1), dtype=torch.float32)
    if not isinstance(emb2, torch.Tensor):
        emb2 = torch.tensor(np.array(emb2), dtype=torch.float32)
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    if emb1.shape != emb2.shape:
        min_len = min(emb1.shape[0], emb2.shape[0])
        emb1 = emb1[:min_len]
        emb2 = emb2[:min_len]
    if emb1.norm().item() == 0 or emb2.norm().item() == 0:
        return 0.0
    sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1)
    return float(sim.item())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train-v2.0.json")
    parser.add_argument("--output_path", type=str, default="data/chunk_feature_label.json")
    parser.add_argument("--embedder", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--target_dim", type=int, default=256)
    args = parser.parse_args()


    print("清洗并准备SQuAD数据...")
    samples = clean_and_prepare_squad(args.data_path)
    print(f"样本数: {len(samples)}")
    all_chunks = []
    chunk_meta = []
    for sample in tqdm(samples, desc="切分chunks", mininterval=1):
        for chunk in sample["chunks"]:
            all_chunks.append(chunk)
            chunk_meta.append({"question": sample["question"], "gold": sample["gold"], "chunk": chunk})

    print(f"总chunk数: {len(all_chunks)}")
    print("加载句向量模型...")
    embedder = SentenceTransformerEmbedder(args.embedder, target_dim=args.target_dim)
    all_chunk_embs = []
    for i in tqdm(range(0, len(all_chunks), 1024), desc="Embedding chunks", mininterval=1):
        batch = all_chunks[i:i+1024]
        batch_embs = embedder.embed_batch(batch)
        if isinstance(batch_embs, torch.Tensor):
            batch_embs = batch_embs.detach().cpu().numpy()
        all_chunk_embs.extend(batch_embs)
    all_chunk_embs = np.vstack(all_chunk_embs)
    all_chunk_embs = torch.from_numpy(all_chunk_embs)

    rag = PureRAG(embedder, llm_generator=None)
    feature_label_list = []

    print("预计算所有问题embedding...")
    question_embs = []
    for meta in tqdm(chunk_meta, desc="问题embedding", mininterval=1):
        question_embs.append(embedder(meta["question"]).detach().cpu().numpy())
    question_embs = np.vstack(question_embs)
    question_embs = torch.from_numpy(question_embs)

    print("开始特征提取与可检索性标注...")
    for idx, meta in enumerate(tqdm(chunk_meta, desc="特征与可检索性标注", mininterval=1)):
        if idx % 100 == 0:
            print(f"已处理: {idx}/{len(chunk_meta)}")
        chunk = meta["chunk"]
        emb = all_chunk_embs[idx].tolist()
        length = len(chunk)
        kw_density = keyword_density(chunk)
        uniq_words = unique_word_count(chunk)
        avg_wlen = avg_word_length(chunk)
        stop_ratio = stopword_ratio(chunk)
        punct_cnt = punctuation_count(chunk)
        num_cnt, year_cnt = number_year_count(chunk)
        sent_cnt = sentence_count(chunk)
        chunk_emb = all_chunk_embs[idx]
        question_emb = question_embs[idx]
        sim_q = embedding_similarity(chunk_emb, question_emb)
        context, details = rag.retrieve(meta["question"], all_chunks, all_chunk_embs, topk=1)
        is_retrievable = (meta["gold"].strip() == context.strip() or meta["gold"] in context)
        feature_label_list.append({
            "embedding": emb,
            "length": length,
            "keyword_density": kw_density,
            "unique_word_count": uniq_words,
            "avg_word_length": avg_wlen,
            "stopword_ratio": stop_ratio,
            "punctuation_count": punct_cnt,
            "number_count": num_cnt,
            "year_count": year_cnt,
            "sentence_count": sent_cnt,
            "chunk_question_similarity": sim_q,
            "is_retrievable": is_retrievable
        })

    print("写入结果文件...")
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(feature_label_list, f, ensure_ascii=False, indent=2)
    print(f"已保存: {args.output_path} (总样本数: {len(feature_label_list)})")

if __name__ == "__main__":
    main()
