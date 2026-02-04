import os
import sys
sys.path.append(os.path.abspath("src"))
import torch
from transformers import AutoTokenizer
import main
TitanMAC = main.TitanMAC
TitanMAG = main.TitanMAG
TitanMAL = main.TitanMAL
TitanModelForLM = main.TitanModelForLM
TitanRAG = main.TitanRAG
import argparse
import json
import re

def generate_answer_label_dataset(data_path, output_path):
    # 参数设置（可根据实际情况调整）
    dim = 1024
    hidden_dim = 1024
    memory_depth = 3
    num_persistent_tokens = 4
    window_size = 256
    threshold = 0.0
    chunk_size = 256
    vocab_size = 32000
    tokenizer_name = "EleutherAI/gpt-neox-20b"
    model_variant = "MAC"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    vocab_size = max(tokenizer.get_vocab().values()) + 1  # 真实embedding大小
    if model_variant == "MAC":
        titan_module = TitanMAC(dim, chunk_size, hidden_dim, memory_depth, num_persistent_tokens, threshold=threshold)
    elif model_variant == "MAG":
        titan_module = TitanMAG(dim, hidden_dim, memory_depth, num_persistent_tokens, window_size, threshold=threshold)
    else:
        titan_module = TitanMAL(dim, hidden_dim, memory_depth, num_persistent_tokens, window_size, threshold=threshold)
    titan_model = TitanModelForLM(titan_module, vocab_size, dim)
    rag = TitanRAG(titan_model)
    rag.to(device)

    with open(data_path, "r", encoding="utf-8") as f:
        squad_data = json.load(f)
    answer_label_data = []
    for article in squad_data.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                # 分词并转tensor
                q_ids = tokenizer(question, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(device)
                c_ids = tokenizer(context, return_tensors="pt", truncation=True, max_length=1024)["input_ids"].to(device)
                # embedding
                q_emb = titan_model.emb(q_ids).unsqueeze(0)  # [1, seq_len, dim]
                c_emb = titan_model.emb(c_ids).unsqueeze(0)
                # RAG推理
                try:
                    rag_answer_emb = rag.query_with_context(q_emb, [c_emb])
                    # 解码为文本
                    rag_answer_ids = torch.argmax(rag_answer_emb, dim=-1)
                    rag_answer = tokenizer.decode(rag_answer_ids[0], skip_special_tokens=True)
                except Exception:
                    rag_answer = None
                if rag_answer and rag_answer.strip():
                    answer_label_data.append({"text": rag_answer.strip(), "label": "可检索"})
                else:
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answer_label_data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answer label dataset for TitanRAG")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data (SQuAD format)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output label dataset")
    args = parser.parse_args()

    generate_answer_label_dataset(args.data_path, args.output_path)