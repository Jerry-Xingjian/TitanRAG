
import sys
import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path to import main (original_benchmarks -> projects -> TitanLLM)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
from main import TitanMAC, TitanMAG, TitanMAL, TitanModelForLM

from associative_recall import AssociativeRecallDataset

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    train_ds = AssociativeRecallDataset(vocab_size=args.vocab_size, seq_len=args.seq_len, num_samples=10000)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
    # Model Setup
    if args.model_variant == "MAC":
        titan_module = TitanMAC(args.dim, args.chunk_size, args.hidden_dim, args.memory_depth, args.num_persistent_tokens, threshold=args.threshold)
    elif args.model_variant == "MAG":
        titan_module = TitanMAG(args.dim, args.hidden_dim, args.memory_depth, args.num_persistent_tokens, args.window_size, threshold=args.threshold)
    else: # MAL
        titan_module = TitanMAL(args.dim, args.hidden_dim, args.memory_depth, args.num_persistent_tokens, args.window_size, threshold=args.threshold)
        
    model = TitanModelForLM(titan_module, args.vocab_size, args.dim)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    model.train()
    
    for epoch in range(args.num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}")
        for input_ids, labels in pbar:
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(input_ids, labels) # TitanModelForLM handles loss calculation if labels provided
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy Metric
            # labels have -100 for ignored positions. We only care about the last token usually.
            # Flatten for argmax
            pred_tokens = torch.argmax(logits, dim=-1)
            mask = labels != -100
            
            correct += (pred_tokens[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct/total:.2%}"})
            
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_variant", type=str, default="MAC", choices=["MAC", "MAG", "MAL"])
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--memory_depth", type=int, default=2)
    parser.add_argument("--num_persistent_tokens", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    
    args = parser.parse_args()
    train(args)
