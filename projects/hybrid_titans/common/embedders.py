"""
Semantic Embedder using sentence-transformers.
Shared by all demo scripts.
"""

import torch
import torch.nn as nn


class SentenceTransformerEmbedder(nn.Module):
    """Semantic embedder using sentence-transformers."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", target_dim=256):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.source_dim = self.model.get_sentence_embedding_dimension()
        self.target_dim = target_dim
        
        if self.source_dim != target_dim:
            self.projection = nn.Linear(self.source_dim, target_dim)
        else:
            self.projection = None
        
        print(f"✅ Loaded SentenceTransformer: {model_name} (dim={self.source_dim})")
    
    def forward(self, text):
        """Embed text into a fixed-size vector."""
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)
            embedding = embedding.clone().cpu()
        
        if self.projection is not None:
            embedding = self.projection(embedding.float())
        
        # Ensure 3D output: [1, seq_len, dim]
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0).unsqueeze(0)
        elif embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)
            
        return embedding
    
    def embed_batch(self, texts):
        """Embed a batch of texts."""
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            embeddings = embeddings.clone().cpu()
        
        if self.projection is not None:
            embeddings = self.projection(embeddings.float())
        
        return embeddings
