# based of the paper "Titans: Learning to Memorize at Test Time"
# https://arxiv.org/pdf/2501.00663v1

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
try:
    import deepspeed
except ImportError:
    deepspeed = None
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from typing import List, Optional
import random

class DeepMemoryModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        # [Simplification for Stability] 
        # Use a single linear layer to guarantee convergence for the demo.
        # Deep MLP + Random Init + Aggressive Online Update = Chaos.
        self.layers.append(nn.Linear(in_dim, out_dim, bias=False)) 
        self.layers.append(nn.Tanh()) # [CRITICAL] Must squash output even for linear! 
        
        # Original Deep MLP logic commented out for strict testing:
        # sizes = [in_dim] + [hidden_dim]*(num_layers-1) + [out_dim]
        # for i in range(num_layers):
        #     self.layers.append(nn.Linear(sizes[i], sizes[i+1], bias=True))
        #     if i < num_layers-1:
        #         self.layers.append(nn.SiLU())
        #     else:
        #         self.layers.append(nn.Tanh())
        self.momentum_buffers = {}
        self.alpha = nn.Parameter(torch.tensor(0.0001)) # Decay rate
        self.theta = nn.Parameter(torch.tensor(0.0001)) # Learning rate
        self.eta = nn.Parameter(torch.tensor(0.9)) # Momentum

    def forward_no_update(self, x):
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def forward_with_update(self, k, v, threshold=None):
        pred = self.forward_no_update(k) # Query memory with key
        diff = (pred - v) # Calculate difference between prediction and target
        # Calculate loss per element/batch to allow fine-grained gating? 
        # For simplicity, we stick to the original aggregate loss but add a threshold check.
        loss = 0.5*(diff**2).sum() # MSE Loss (Surprise)
        
        # [NEW] Sparse Update Logic
        # If the model predicts well, loss is small; if poorly, loss is large
        if threshold is not None:
            # If surprise is below threshold, it means "this info is not worth remembering", skip expensive gradient calculation.
            if loss.item() < threshold:
                if random.random() < 0.01: # Log 1% of skips to verify
                    print(f"Skipping update (Surprise={loss.item():.4f} < {threshold})")
                return

        # [Fix] Safety Check for Inference Mode (no_grad)
        # If we are in no_grad mode, loss has no grad_fn, so autograd.grad will fail.
        if not loss.requires_grad:
            return

        # Manually call autograd.grad to calculate gradients during forward pass. This is "Test-Time Training".
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=False, allow_unused=True)
        idx = 0
        # Update rule similar to SGD+Momentum, but executed during inference. alpha controls forgetting, theta controls learning, eta controls momentum.
        with torch.no_grad():
            for param in self.parameters():
                # 1. Calculate gradient (g): This represents "How should my neurons change if I want to memorize this surprising input?"
                g = grads[idx]
                idx += 1
                if g is None:
                    continue
                
                # [Fix] Gradient Clipping to prevent NaN/Explosion
                g = torch.clamp(g, -1.0, 1.0)
                
                if not hasattr(param, "surprise"):
                    param.surprise = torch.zeros_like(param.data)
                
                # 2. Calculate momentum (Surprise State)
                new_surprise = (self.eta*param.surprise) - (self.theta*g)
                # [Fix] Clamp Surprise State
                param.surprise = torch.clamp(new_surprise, -1.0, 1.0)
                
                # 3. Actual weight update (Weight Update)
                new_data = (1 - self.alpha)*param.data + param.surprise
                # [Fix] Clamp Weights
                param.data = torch.clamp(new_data, -2.0, 2.0)

    def forward(self, x):
        return self.forward_no_update(x)


# ============================================================================
# Persistent Memory
# ============================================================================
class PersistentMemory(nn.Module):
    """
    Learned, fixed tokens that are prepended to the input sequence.
    Acts as a learnable "global context" or "system prompt" for the model.
    """
    # A set of fixed learnable parameters, prepended to every input sequence as a "global context prefix". Similar to Prompt Tuning.
    def __init__(self, num_tokens, dim):
        super().__init__()
        # Parameter: [num_tokens, dim], shared across all batches
        self.mem = nn.Parameter(torch.randn(num_tokens, dim))

    def forward(self, bsz):
        """
        Expands the persistent memory for the current batch size.
        Returns: [bsz, num_tokens, dim]
        """
        return self.mem.unsqueeze(0).expand(bsz, -1, -1)

# ============================================================================
# Attention Modules
# ============================================================================
class SlidingWindowAttention(nn.Module):
    """
    Local Attention mechanism that only attends to tokens within a fixed window size.
    Reduces computational complexity from O(N^2) to O(N * window_size).
    Used in MAG and MAL variants as the "Short-Term Memory".
    """
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seq_len, d = x.size()
        h = self.num_heads
        
        # Calculate Q, K, V
        q = self.wq(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        k = self.wk(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        v = self.wv(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        
        out_chunks = []
        # Naive implementation of Sliding Window (for clarity, not speed optimization)
        for i in range(seq_len):
            # Define window start index
            start = max(0, i - self.window_size + 1)
            
            # Extract Q for current token, K and V for the window
            q_i = q[:, :, i:i+1, :]
            k_i = k[:, :, start:i+1, :] # Look only at Key within window
            v_i = v[:, :, start:i+1, :]
            
            # Standard Attention Computation
            # Standard Attention is O(n^2), here changed to only attend to the nearest window_size Tokens, reducing computational complexity. Used for MAG/MAL architecture.
            logits = torch.matmul(q_i, k_i.transpose(-1, -2)) / math.sqrt(d//h)
            attn = F.softmax(logits, dim=-1)
            val = torch.matmul(attn, v_i)
            out_chunks.append(val)
            
        out = torch.cat(out_chunks, dim=2).transpose(1,2).reshape(bsz, seq_len, d)
        out = self.out(out)
        return out

class FullAttention(nn.Module):
    """
    Standard Scaled Dot-Product Attention mechanism.
    Complexity: O(N^2).
    Used in MAC variant within small chunks.
    """
    # Standard Scaled Dot-Product Attention, used for MAC architecture (since MAC is already chunked, full attention is used within each chunk).
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seq_len, d = x.size()
        h = self.num_heads
        
        # Calculate Q, K, V
        q = self.wq(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        k = self.wk(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        v = self.wv(x).view(bsz, seq_len, h, d//h).transpose(1,2)
        
        # Compute Attention Scores
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d//h)
        attn = F.softmax(logits, dim=-1)
        
        # Apply Attention to V
        val = torch.matmul(attn, v)
        out = val.transpose(1,2).reshape(bsz, seq_len, d)
        out = self.out(out)
        return out


# ============================================================================
# Titan Model Variants
# ============================================================================

class TitanMAC(nn.Module):
    """
    TitanMAC (Memory As Context)
    
    Structure:
    1. Segment input into chunks.
    2. For each chunk:
       a. Retrieve context from Long-Term Memory (LTM).
       b. Concatenate [Persistent Memory, Retrieved Context, Current Chunk].
       c. Process concatenated sequence with Full Attention.
       d. Update LTM using the current chunk (Online Learning).
    
    Key Idea: LTM provides historical context which is simply prepended to the 
    current context window, similar to RAG or Transformer-XL but with learned memory.
    """
    def __init__(self, dim, chunk_size, hidden_dim, memory_depth, num_persistent_tokens, threshold=0.0):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        self.threshold = threshold
        # Core Neural Memory Module (Long-term Memory)
        self.ltm = DeepMemoryModule(dim, hidden_dim, dim, memory_depth)
        # Fixed Persistent Context (Persistent Memory)
        self.pm = PersistentMemory(num_persistent_tokens, dim)
        # Short-term processing within the chunk (Attention)
        self.attn = FullAttention(dim, 4)

    def segment(self, x):
        """Helper to split input sequence into chunks."""
        bsz, seq_len, d = x.size()
        chunks = []
        idx = 0
        while idx < seq_len:
            chunks.append(x[:, idx:idx+self.chunk_size, :])
            idx += self.chunk_size
        return chunks

    # Read memory first, then calculate Attention, finally use current chunk output to update memory
    def forward(self, x):
        bsz, seq_len, d = x.size()
        out_chunks = []
        segs = self.segment(x) #  Split sequence into multiple Chunks
        
        for sg in segs:
            # 1. Retrieve from Memory (Forward Pass without Update)
            # Retrieve history from memory
            q2d = sg.reshape(-1, d)
            with torch.no_grad():
                ret2d = self.ltm.forward_no_update(q2d)
            ret = ret2d.view(sg.size())
            
            # 2. Concatenate Contexts
            p = self.pm(bsz)
            # Concatenate: [Persistent Memory, Historical Memory, Current Chunk]
            cat_in = torch.cat([p, ret, sg], dim=1)
            
            # 3. Process with Attention
            out_attn = self.attn(cat_in)
            chunk_out = out_attn[:, -sg.size(1):, :] # Extract only the outputs corresponding to input
            
            # 4. Update Memory (Test-Time Training)
            k2d = sg.reshape(-1, d)
            v2d = chunk_out.reshape(-1, d)
            # This step updates LTM parameters based on the prediction error
            self.ltm.forward_with_update(k2d, v2d, threshold=self.threshold)
            
            out_chunks.append(chunk_out)
        return torch.cat(out_chunks, dim=1)

class TitanMAG(nn.Module):
    """
    TitanMAG (Memory As Gating)
    
    Structure:
    1. Two branches process the input in parallel:
       a. Sliding Window Attention (Short-Term).
       b. DeepMemoryModule (Long-Term).
    2. Outputs are merged via a learned gating mechanism.
    3. LTM is updated at every step (or gated by threshold).
    
    Key Idea: Soft fusion of short-term and long-term representations.
    """
    def __init__(self, dim, hidden_dim, memory_depth, num_persistent_tokens, window_size, threshold=0.0):
        super().__init__()
        self.dim = dim
        self.threshold = threshold
        self.pm = PersistentMemory(num_persistent_tokens, dim)
        self.swa = SlidingWindowAttention(dim, 4, window_size)
        self.ltm = DeepMemoryModule(dim, hidden_dim, dim, memory_depth)
        # Gating projections
        self.lin1 = nn.Linear(dim, dim, bias=False)
        self.lin2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seq_len, d = x.size()
        pm_out = self.pm(bsz)
        x_in = torch.cat([pm_out, x], dim=1)
        
        # Branch 1: Short-term Memory (Attention)
        # Short-term: Sliding Window Attention
        out_swa = self.swa(x_in)
        
        # Branch 2: Long-term Memory
        ltm_in = x_in.reshape(-1, d)
        
        # Update Memory first (Note: In MAG implementation here, update happens concurrently)
        # Using SWA output as the target for memory update
        # Long-term: Update Memory
        self.ltm.forward_with_update(ltm_in, out_swa.reshape(-1, d), threshold=self.threshold)
        
        # Retrieve Memory context
        # Read Memory
        out_mem = self.ltm.forward_no_update(ltm_in).view(*x_in.shape)
        
        # Gating and Fusion
        g1 = torch.sigmoid(self.lin1(out_swa))
        g2 = torch.sigmoid(self.lin2(out_mem))
        out_merged = g1*out_swa + g2*out_mem
        
        return out_merged[:, pm_out.size(1):, :]

class TitanMAL(nn.Module):
    """
    TitanMAL (Memory As Layer)
    
    Structure:
    1. Input tokens act as queries to the LTM sequentially.
    2. LTM processes tokens and updates itself step-by-step.
    3. The sequence of LTM outputs is then fed into Sliding Window Attention.
    
    Key Idea: Memory acts as a "pre-processing layer" that enriches tokens 
    with historical context before they reach the attention mechanism.
    """
    # Similar to RNN, the memory module acts as a "pre-processing layer", updating memory token by token, then passing through Attention.
    def __init__(self, dim, hidden_dim, memory_depth, num_persistent_tokens, window_size, threshold=0.0):
        super().__init__()
        self.threshold = threshold
        self.pm = PersistentMemory(num_persistent_tokens, dim)
        self.ltm = DeepMemoryModule(dim, hidden_dim, dim, memory_depth)
        self.swa = SlidingWindowAttention(dim, 4, window_size)

    def forward(self, x):
        bsz, seq_len, d = x.size()
        pm_out = self.pm(bsz)
        x_cat = torch.cat([pm_out, x], dim=1)
        out_ltm = []
        
        # Sequential Processing (Recurrent-like)
        # Token-by-Token Loop
        for i in range(x_cat.size(1)):
            token = x_cat[:, i:i+1, :]
            k2d = token.view(-1, d)
            
            # Predict/Retrieve
            pred = self.ltm.forward_no_update(k2d)
            
            # Update Memory with current token
            # Update memory at every Token
            self.ltm.forward_with_update(k2d, pred, threshold=self.threshold)
            
            # [Fix] Residual Connection: x + Memory(x)
            # Without this, the memory module acts as a bottleneck/scrambler (if untrained).
            # This allows gradients/info to flow through even if Memory is garbage.
            res_out = pred + k2d
            
            out_ltm.append(res_out.view(bsz, 1, d))
            
        out_ltm = torch.cat(out_ltm, dim=1)
        
        # Post-process with Attention
        # Finally pass through Attention
        out_swa = self.swa(out_ltm)
        return out_swa[:, pm_out.size(1):, :]

class TitanDataset(Dataset):
    def __init__(self, split, tokenizer_name, seq_len=1024):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.seq_len = seq_len
        self.samples = []
        if split=="train":
            pass
        else:
            pass
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        return torch.randint(0, 32000, (self.seq_len,)), torch.randint(0, 32000, (self.seq_len,))

def titan_collate(batch):
        xs, ys = [], []
        for x,y in batch:
            xs.append(x)
            ys.append(y)
        return torch.stack(xs,dim=0), torch.stack(ys,dim=0)

class TitanModelForLM(nn.Module):
    def __init__(self, titan_module, vocab_size, dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.titan = titan_module
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.emb(input_ids)
        y = self.titan(x)
        logits = self.head(y)
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return logits, loss
        return logits, None

class TitanRAG(nn.Module):
    """
    TitanRAG: A Hybrid Neuro-Symbolic Architecture wrapper.
    enables 'Test-Time Training' based retrieval augmentation.
    """
    def __init__(self, titan_model):
        super().__init__()
        self.titan = titan_model
        
    def forward(self, x):
        return self.titan(x)

    def digest_knowledge(self, document_embeddings):
        """
        The Core Innovation: "Learning" the retrieved documents on-the-fly.
        Args:
            document_embeddings (torch.Tensor): [1, doc_len, dim]
        """
        # 1. Segment the document
        if hasattr(self.titan, 'segment'):
            chunks = self.titan.segment(document_embeddings)
        else:
            chunks = [document_embeddings]

        # 2. Test-Time Training loop
        for i, chunk in enumerate(chunks):
            bsz, seq_len, dim = chunk.size()
            
            # Auto-associative update (Self-Supervised)
            # Treating the chunk as both input (Key) and target (Value)
            k2d = chunk.reshape(-1, dim)
            v2d = chunk.reshape(-1, dim)
            
            # This updates the LTM weights immediately!
            # Uses the model's configured threshold to skip known info.
            self.titan.ltm.forward_with_update(k2d, v2d, threshold=self.titan.threshold)
            
    def query_with_context(self, query_embeddings, retrieved_docs_embeddings):
        """
        Full RAG Pipeline: Retrieve -> Digest -> Query
        """
        # Step 1: Digestion ("Reading" Phase)
        for doc in retrieved_docs_embeddings:
            self.digest_knowledge(doc)
            
        # Step 2: Inference ("Answering" Phase)
        # Context is CLEAN (just the query), weights are ENRICHED.
        return self.titan(query_embeddings)


def get_ds_config(args):
    return {
        "train_batch_size": args.global_batch_size,
        "gradient_accumulation_steps": args.grad_acc,
        "fp16": {
            "enabled": args.fp16
        },
        "zero_optimization": {
            "stage": args.zero_stage
        },
        "zero_allow_untested_optimizer": True
    }

def train_main(args):
    dist.init_process_group("nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    train_ds = TitanDataset("train", args.tokenizer_name, seq_len=args.seq_len)
    val_ds = TitanDataset("val", args.tokenizer_name, seq_len=args.seq_len)
    train_sampler = RandomSampler(train_ds) if not dist.is_initialized() else torch.utils.data.distributed.DistributedSampler(train_ds)
    val_sampler = SequentialSampler(val_ds)
    train_dl = DataLoader(train_ds, sampler=train_sampler, batch_size=args.batch_size, collate_fn=titan_collate, drop_last=True, num_workers=2)
    val_dl = DataLoader(val_ds, sampler=val_sampler, batch_size=args.batch_size, collate_fn=titan_collate, drop_last=False, num_workers=2)

    if args.model_variant=="MAC":
        titan_module = TitanMAC(args.dim, args.chunk_size, args.hidden_dim, args.memory_depth, args.num_persistent_tokens, threshold=args.threshold)
    elif args.model_variant=="MAG":
        titan_module = TitanMAG(args.dim, args.hidden_dim, args.memory_depth, args.num_persistent_tokens, args.window_size, threshold=args.threshold)
    else:
        titan_module = TitanMAL(args.dim, args.hidden_dim, args.memory_depth, args.num_persistent_tokens, args.window_size, threshold=args.threshold)
    model = TitanModelForLM(titan_module, args.vocab_size, args.dim)

    ds_config = get_ds_config(args)
    engine, optimizer, _, scheduler = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

    global_steps = 0
    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dl):
            inp, lab = batch
            inp = inp.cuda()
            lab = lab.cuda()
            logits, loss = engine(inp, lab)
            engine.backward(loss)
            engine.step()
            global_steps += 1
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_count = 0
            for batch in val_dl:
                inp, lab = batch
                inp = inp.cuda()
                lab = lab.cuda()
                logits, loss = engine(inp, lab)
                total_loss += loss.item() * inp.size(0)
                total_count += inp.size(0)
            val_ppl = math.exp(total_loss/total_count)
    dist.destroy_process_group()

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--global_batch_size", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--memory_depth", type=int, default=3)
    parser.add_argument("--num_persistent_tokens", type=int, default=4)
    parser.add_argument("--model_variant", type=str, default="MAC")
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = get_args()
    train_main(args)
