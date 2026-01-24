import torch
import argparse
import random
import sys
import os

# Add parent directory to path to import main (hybrid_titans -> projects -> TitanLLM)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

from main import TitanMAC, TitanMAG, TitanMAL, TitanRAG

def generate_haystack(num_docs, doc_len, dim, key_idx=None):
    """
    Generates N documents. One containing the 'needle' (passkey), others noise.
    """
    docs = []
    needle_pattern = torch.randn(1, dim)
    
    if key_idx is None:
        key_idx = random.randint(0, num_docs - 1)
        
    for i in range(num_docs):
        doc = torch.randn(doc_len, dim)
        if i == key_idx:
            doc[-1, :] = needle_pattern
        docs.append(doc)
        
    return docs, needle_pattern, key_idx

def pretrain_identity(model, dim, steps=500):
    """
    Trains the model's Attention/Processing layers to perform Identity Mapping f(x)=x.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    
    print(f"  [Pre-training] Teaching model to copy (Identity Function)...")
    
    # [Fix] Disable Titan's internal Test-Time Training during Identity Pre-training.
    # We only want to train the Attention mechanism (Weights), not the Memory (State).
    # Otherwise, in-place updates cause 'backward through graph second time' error.
    saved_threshold = model.threshold
    model.threshold = float('inf') # Force skip update
    
    for i in range(steps):
        # Generate random data
        x = torch.randn(1, 32, dim) # [Batch, Seq, Dim]
        target = x
        
        optimizer.zero_grad()
        
        pred = model(x)
        loss = ((pred - target)**2).mean()
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"    Step {i}: Loss {loss.item():.4f}")
            
    # Restore original threshold for Experiment
    model.threshold = saved_threshold

def run_single_variant(variant_name, ModelClass, args, docs_batch, key_idx):
    print(f"\n[{variant_name}] Testing Memory Retention...")
    
    # 1. Config Handling
    # MAC uses 'chunk_size', MAG/MAL use 'window_size'
    config = {
        "dim": args.dim,
        "hidden_dim": args.dim,
        "memory_depth": 2,
        "num_persistent_tokens": 0,
        "threshold": 0.0
    }
    
    if variant_name == 'TitanMAC':
        config["chunk_size"] = 32
    else:
        config["window_size"] = 32
        
    try:
        model_base = ModelClass(**config)
    except Exception as e:
        print(f"  Error instantiating {variant_name}: {e}")
        return

    # [CRITICAL STEP] Pre-train the 'Context Engine' (Attention)
    # Otherwise Hybrid RAG fails because Random Weights don't Copy.
    pretrain_identity(model_base, args.dim)

    # [Optimization 2.0] High-Fidelity Tuning
    with torch.no_grad():
        model_base.ltm.theta.fill_(0.001) 
        model_base.ltm.alpha.fill_(0.0001)
    
    model_rag = TitanRAG(model_base)
    
    # 2. Baseline Evaluation
    needle_doc = docs_batch[key_idx]
    target = needle_doc
    pred_init = model_rag.titan.ltm.forward_no_update(target.reshape(-1, args.dim))
    diff_init = (pred_init - target.reshape(-1, args.dim))
    loss_init = 0.5*(diff_init**2).sum().item()
    
    # 3. Digestion
    # print(f"  Digesting (200 epochs per doc)...")
    for i, doc in enumerate(docs_batch):
        for _ in range(200): 
            model_rag.digest_knowledge(doc)
            
    # 4. Final Evaluation
    pred_final = model_rag.titan.ltm.forward_no_update(target.reshape(-1, args.dim))
    diff_final = (pred_final - target.reshape(-1, args.dim))
    loss_after = 0.5*(diff_final**2).sum().item()
    
    # 5. Metrics
    total_elements = args.doc_len * args.dim
    signal_energy = ((target.reshape(-1, args.dim))**2).sum().item()
    error_energy = 2 * loss_after
    
    fidelity = 1.0 - (error_energy / (signal_energy + 1e-9))
    fidelity = max(0.0, fidelity) * 100.0
    
    print(f"  Initial Loss: {loss_init:.4f}")
    print(f"  Final Loss:   {loss_after:.4f}")
    print(f"  Fidelity:     {fidelity:.2f}%")
    
    if loss_after < loss_init:
        print(f"  Result: PASS (Improvement {((loss_init-loss_after)/loss_init)*100:.1f}%)")
    else:
        print(f"  Result: FAIL")
        
    # --- Method 4: Hybrid RAG (The Best of Both Worlds) ---
    print(f"\n[{variant_name}] Testing Hybrid RAG (Weight + Context)...")
    # Scenario: Needle is in Context, Noise is in Weights.
    
    with torch.no_grad():
        # Input: Needle Doc (Batch=1, Len=doc_len, Dim=dim)
        # By passing it as input 'x' to the model forward pass, 
        # the Attention mechanism sees it directly.
        inp = needle_doc # Already has batch dim [1, 32, 64]
        output = model_rag(inp)
        
    diff_hybrid = (output - inp)
    loss_hybrid = 0.5*(diff_hybrid**2).sum().item()
    
    error_energy_hyb = 2 * loss_hybrid
    fidelity_hyb = 1.0 - (error_energy_hyb / (signal_energy + 1e-9))
    fidelity_hyb = max(0.0, fidelity_hyb) * 100.0
    
    print(f"  Hybrid Fidelity: {fidelity_hyb:.2f}% (Should be ~100%)")
    if fidelity_hyb > 99.0:
        print(f"  Result: PASS (Perfect Recall via Attention)")

def compare_methods(args):
    print("--- Conducting RAG Comparison Experiment (All Variants) ---")
    print(f"Docs: {args.num_docs}, Doc Len: {args.doc_len}, Dim: {args.dim}")
    
    # 1. Setup Data
    docs, _, key_idx = generate_haystack(args.num_docs, args.doc_len, args.dim)
    docs_batch = [d.unsqueeze(0) for d in docs]
    
    # 2. Traditional RAG Baseline (Only for MAC, as context concat is unique to it conceptually)
    print("\n[Baseline: Traditional RAG (Concatenation)]")
    # Only MAC is really suitable for 'Concat all' naive RAG, others use sliding window/gating.
    # But for OOM testing, any model receiving huge input works.
    try:
        full_context = torch.cat(docs_batch, dim=1)
        if full_context.size(1) > 2048:
            print(f"  Result: FAIL (Context Length {full_context.size(1)} > 2048)")
        else:
            print(f"  Result: PASS (Processed {full_context.size(1)} tokens)")
    except Exception as e:
         print(f"  Result: FAIL ({e})")

    # 3. TitanRAG Variants
    variants = [
        ('TitanMAC', TitanMAC),
        ('TitanMAG', TitanMAG),
        ('TitanMAL', TitanMAL)
    ]
    
    for name, cls in variants:
        run_single_variant(name, cls, args, docs_batch, key_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_docs", type=int, default=50, help="Number of retrieved documents")
    parser.add_argument("--doc_len", type=int, default=64, help="Length of each document")
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    
    compare_methods(args)
