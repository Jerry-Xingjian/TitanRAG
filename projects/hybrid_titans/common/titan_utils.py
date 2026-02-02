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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    
    device_name = "GPU" if device.type == "cuda" else "CPU"
    print(f"✅ Created TitanRAG with {arch} architecture (dim={dim}) on {device_name}")
    
    return titan_rag


def digest_document(titan_rag, embedder, text, epochs=50):
    """
    Digest a document into TitanRAG memory.
    
    Args:
        titan_rag: TitanRAG instance
        embedder: Embedder to convert text to embeddings
        text: Document text (will be split by paragraphs)
        epochs: Number of digestion epochs
    
    Returns:
        tuple: (paragraphs, paragraph_embeddings)
    """
    # Parse into paragraphs
    raw_paragraphs = text.split('\n\n')
    paragraphs = []
    for para in raw_paragraphs:
        cleaned = ' '.join(line.strip() for line in para.split('\n') if line.strip())
        if cleaned and len(cleaned) > 10:
            paragraphs.append(cleaned)
    
    # Embed paragraphs
    all_embeddings = torch.cat([embedder(p) for p in paragraphs], dim=1)
    
    # Digest into memory
    print(f"🧠 Digesting {len(paragraphs)} paragraphs ({epochs} epochs)...")
    for epoch in range(epochs):
        titan_rag.digest_knowledge(all_embeddings)
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{epochs} completed")
    
    # Compute paragraph embeddings for retrieval
    paragraph_embeddings = embedder.embed_batch(paragraphs)
    
    return paragraphs, paragraph_embeddings
