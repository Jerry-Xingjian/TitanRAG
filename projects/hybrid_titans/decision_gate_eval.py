import os
import sys
import argparse
import random
import math
import torch
import torch.nn.functional as F

# Ensure repo src is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

from main import TitanMAG, TitanRAG


def generate_haystack(num_docs, doc_len, dim, key_idx=None):
    docs = []
    needle_pattern = torch.randn(1, dim)
    if key_idx is None:
        key_idx = random.randint(0, num_docs - 1)
    for i in range(num_docs):
        doc = torch.randn(1, doc_len, dim)
        if i == key_idx:
            doc[0, -1, :] = needle_pattern
        docs.append(doc)
    return docs, needle_pattern, key_idx


def pretrain_identity(model, dim, steps=200):
    # small pretrain to make attention behave like identity mapping
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    saved_threshold = getattr(model, 'threshold', None)
    if saved_threshold is not None:
        model.threshold = float('inf')
    for i in range(steps):
        x = torch.randn(1, 32, dim)
        opt.zero_grad()
        out = model(x)
        loss = ((out - x)**2).mean()
        loss.backward()
        opt.step()
    if saved_threshold is not None:
        model.threshold = saved_threshold


def mse_loss(a, b):
    return 0.5 * ((a - b)**2).mean().item()


def fidelity_from_loss(target, loss):
    # simple proxy: fidelity = 1 - error_energy / signal_energy
    signal_energy = (target**2).mean().item()
    error_energy = 2 * loss
    fidelity = 1.0 - (error_energy / (signal_energy + 1e-9))
    return max(0.0, fidelity) * 100.0


def run_compare(args):
    print('Generating haystack...')
    docs, needle_pattern, key_idx = generate_haystack(args.num_docs, args.doc_len, args.dim)

    # Build model
    config = {
        'dim': args.dim,
        'hidden_dim': args.dim,
        'memory_depth': 2,
        'num_persistent_tokens': 0,
        'threshold': 0.0,
        'window_size': 32
    }
    model = TitanMAG(**config)
    pretrain_identity(model, args.dim, steps=100)
    with torch.no_grad():
        model.ltm.theta.fill_(0.001)
        model.ltm.alpha.fill_(0.0001)

    rag = TitanRAG(model)

    # Target doc
    target = docs[key_idx]

    # Mode A: Titan-only (digest target many times, then evaluate reconstruction)
    titan_only_model = TitanRAG(TitanMAG(**config))
    pretrain_identity(titan_only_model.titan, args.dim, steps=100)
    with torch.no_grad():
        titan_only_model.titan.ltm.theta.fill_(0.001)
        titan_only_model.titan.ltm.alpha.fill_(0.0001)

    # digest target
    for _ in range(args.digest_steps):
        titan_only_model.digest_knowledge(target)

    # Evaluate Titan-only recall
    flat = target.reshape(-1, args.dim)
    pred = titan_only_model.titan.ltm.forward_no_update(flat)
    loss_titan = mse_loss(pred, flat)
    fid_titan = fidelity_from_loss(flat, loss_titan)

    # Mode B: Full Hybrid (pass target as input to model forward)
    # We use rag.titan forward to simulate attention+memory usage (hybrid)
    with torch.no_grad():
        out = rag.titan(target)
    loss_hybrid = mse_loss(out, target)
    fid_hybrid = fidelity_from_loss(target, loss_hybrid)

    # Mode C: Decision-gated
    # Compute features: retr_score (simulated by cosine between query and doc), mem_sim and surprise
    # For this demo, use last token as 'query'
    query = target[:, -1:, :].reshape(-1, args.dim)
    # retr_score: cosine between query and doc mean
    doc_vec = target.mean(dim=1).reshape(1, -1)
    query_vec = query.mean(dim=0, keepdim=True)
    retr_score = F.cosine_similarity(query_vec, doc_vec).item()

    # memory prediction
    mem_pred = rag.titan.ltm.forward_no_update(query)
    mem_vec = mem_pred.mean(dim=0, keepdim=True)
    mem_sim = F.cosine_similarity(mem_vec, query_vec).item()

    # surprise: MSE between mem_pred and query
    surprise = 0.5 * ((mem_pred - query)**2).mean().item()

    # Decide gate g
    w1, w2, w3, b = 1.0, 0.8, 1.5, -0.2
    g_scalar = torch.sigmoid(torch.tensor(w1*retr_score + w2*mem_sim - w3*surprise + b)).item()

    # Get context branch output (attention on context). For simplicity, use rag.titan forward on target as context_out
    with torch.no_grad():
        context_out = rag.titan(target)
        # memory branch output: replicate mem_pred across sequence length
        mem_out_seq = rag.titan.ltm.forward_no_update(target.reshape(-1, args.dim)).view(target.size())

    out_gate = g_scalar * context_out + (1.0 - g_scalar) * mem_out_seq
    loss_gate = mse_loss(out_gate, target)
    fid_gate = fidelity_from_loss(target, loss_gate)

    print('\nResults:')
    print(f'  Titan-only fidelity: {fid_titan:.2f}%, loss: {loss_titan:.6f}')
    print(f'  Hybrid (full) fidelity: {fid_hybrid:.2f}%, loss: {loss_hybrid:.6f}')
    print(f'  Decision-gated fidelity: {fid_gate:.2f}%, loss: {loss_gate:.6f}, gate={g_scalar:.3f}, retr={retr_score:.3f}, mem_sim={mem_sim:.3f}, surprise={surprise:.6f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_docs', type=int, default=20)
    parser.add_argument('--doc_len', type=int, default=32)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--digest_steps', type=int, default=200)
    args = parser.parse_args()
    run_compare(args)


if __name__ == '__main__':
    main()
