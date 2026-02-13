"""
Titan model utilities.
Shared initialization and configuration for TitanMAG/TitanRAG.
"""

import torch
import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'src'))

from main import TitanMAG, TitanMAC, TitanRAG

# Detect device
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    try:
        import torch_xla.core.xla_model as xm
        DEVICE = xm.xla_device()
        print(f"✅ TPU detected: using XLA device {DEVICE}")
    except ImportError:
        pass


def create_titan_rag(dim=256, arch="MAG", window_size=32, memory_depth=3, 
                     num_persistent_tokens=4, threshold=0.0,
                     learning_rate=0.01, forgetting_rate=0.0001,
                     device=None):
    """
    Create and configure a TitanRAG instance.
    
    Args:
        dim: Embedding dimension
        arch: Architecture type ("MAG" or "MAC")
        window_size: Sliding window size for attention
        memory_depth: Number of layers in memory MLP
        num_persistent_tokens: Number of persistent memory tokens
        threshold: Surprise threshold for sparse updates
        learning_rate: Learning rate (theta) for memory updates
        forgetting_rate: Forgetting rate (alpha) for memory decay
        device: Device to use (default: auto-detect)
    
    Returns:
        TitanRAG: Configured TitanRAG instance
    """
    if device is None:
        device = DEVICE
    
    if arch.upper() == "MAC":
        base_model = TitanMAC(
            dim=dim,
            chunk_size=window_size,
            hidden_dim=dim * 2,
            memory_depth=memory_depth,
            num_persistent_tokens=num_persistent_tokens,
            threshold=threshold
        )
    else:
        base_model = TitanMAG(
            dim=dim,
            window_size=window_size,
            hidden_dim=dim * 2,
            memory_depth=memory_depth,
            num_persistent_tokens=num_persistent_tokens,
            threshold=threshold
        )
    
    # Move model to device
    base_model = base_model.to(device)
    
    # Configure memory parameters
    with torch.no_grad():
        base_model.ltm.theta.fill_(learning_rate)
        base_model.ltm.alpha.fill_(forgetting_rate)
    
    titan_rag = TitanRAG(base_model)
    
    if str(device).startswith('xla'):
        device_name = "TPU"
    elif device.type == "cuda":
        device_name = "GPU"
    else:
        device_name = "CPU"
    print(f"✅ Created TitanRAG with {arch} architecture (dim={dim}) on {device_name}")
    
    return titan_rag


def digest_chunks(embedder, titan_rag, chunks, epochs, inline=False):
    """Digest chunks into Titan memory.

    Args:
        embedder: SentenceTransformerEmbedder instance
        titan_rag: TitanRAG instance
        chunks: List of text chunks
        epochs: Max digestion epochs (capped at 500)
        inline: If True, print progress inline ("10 20 30 done").
                If False, print epoch lines.
    """
    digest_epochs = min(epochs, 500)
    if inline:
        print(f"   Digesting ({digest_epochs} epochs): ", end="", flush=True)
    else:
        print(f"\n🧠 Digesting {len(chunks)} chunks into shared memory ({digest_epochs} epochs)...")

    # Pre-compute embeddings once (chunks don't change between epochs)
    cached_emb = embedder.embed_batch(chunks).unsqueeze(0)

    for epoch in range(digest_epochs):
        titan_rag.digest_knowledge(cached_emb)
        if (epoch + 1) % 10 == 0:
            if inline:
                print(f"{epoch+1}", end=" ", flush=True)
            else:
                print(f"   Epoch {epoch + 1}/{digest_epochs}")

    if inline:
        print("done")

