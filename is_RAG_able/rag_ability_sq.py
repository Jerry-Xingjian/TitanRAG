import json
import torch
import argparse
import os
import sys
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from main import TitanMAG, TitanRAG

def run_essay_experiment(args):
    # 1. Load Embedder and TitanRAG model
    print("Loading embedder and TitanRAG model...")
    # Initialize embedder (simple character-level, same as essay_rag_demo)
    class SimpleTextEmbedder(nn.Module):
        def __init__(self, dim=256, vocab_size=256):
            super().__init__()
            self.dim = dim
            self.embed = nn.Embedding(vocab_size, dim)
        def forward(self, text):
            bytes_list = [min(ord(c), 255) for c in text]
            tokens = torch.tensor(bytes_list, dtype=torch.long)
            emb = self.embed(tokens)
            return emb.mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,dim]

    embedder = SimpleTextEmbedder(dim=args.dim)

    # We will NOT use Titan memory for this experiment.
    # Retrieval will be done directly via cosine similarity over context sentence embeddings.
    titan_rag = None

    # 2. Load SQuAD v2.0 Training Data (Questions + Context Only)
    print("\n📂 Loading SQuAD training data...")
    with open("/Users/jerry/Desktop/Research Class/TitanRAG/data/train-v2.0.json", "r", encoding="utf-8") as f:
        squad = json.load(f)

    # Collect (context, question) pairs
    samples = []
    for article in squad["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                answers = [a["text"] for a in qa.get("answers", []) if a["text"].strip()]
                samples.append((context, question, answers))

    print(f"Loaded {len(samples)} question-context pairs")

    # 3. Use first context as the document to digest
    context_text, _, _ = samples[0]
    context_lines = [
        line.strip() for line in context_text.split('.')
        if line.strip() and len(line.strip()) > 20
    ]
    print(f"Context split into {len(context_lines)} sentences")

    # 4. Embed each context line separately
    line_embeddings = []
    with torch.no_grad():
        for line in context_lines:
            emb = embedder(line)
            line_embeddings.append(emb.reshape(-1, args.dim).mean(dim=0))
    line_embeddings = torch.stack(line_embeddings)

    # 5. No Titan digestion — pure embedding-based retrieval baseline
    print("\n🧠 Skipping Titan memory digestion (pure RAG baseline)...")

    # 6. Retrieval-only Question Evaluation
    print("\n🔍 Retrieval-only Question Evaluation\n")
    for i, (_, question, gold_answers) in enumerate(samples[:20], 1):
        query_emb = embedder(question)
        query_vec = query_emb.reshape(-1, args.dim).mean(dim=0)

        sims = torch.nn.functional.cosine_similarity(
            query_vec.unsqueeze(0), line_embeddings
        )
        topk_vals, topk_idxs = sims.topk(min(args.topk, len(context_lines)))

        print(f"Q{i}: {question}")
        retrieved_sentences = [context_lines[idx] for idx in topk_idxs.tolist()]

        print("  Per-sentence RAG-ability (Top-k):")
        sentence_labels = []
        for rank, sent in enumerate(retrieved_sentences, 1):
            match = any(ans.lower() in sent.lower() for ans in gold_answers)
            label = "RAG-able" if match else "Non-RAG-able"
            sentence_labels.append(label)
            print(f"    {rank}. {label} | {sent[:150]}...")

        print(f"  Gold Answers: {gold_answers[:3]}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--dim", type=int, default=256)

    args = parser.parse_args()
    run_essay_experiment(args)
