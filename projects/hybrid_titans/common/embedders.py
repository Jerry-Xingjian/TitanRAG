"""
Semantic Embedder using sentence-transformers.
Shared by all demo scripts.
"""

import torch
import torch.nn as nn


# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentenceTransformerEmbedder(nn.Module):
    """Semantic embedder using sentence-transformers."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2", target_dim=256, device=None):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        
        if device is None:
            device = DEVICE
        self.device = device
        
        self.model = SentenceTransformer(model_name, device=str(device))
        self.source_dim = self.model.get_sentence_embedding_dimension()
        self.target_dim = target_dim
        
        if self.source_dim != target_dim:
            self.projection = nn.Linear(self.source_dim, target_dim).to(device)
        else:
            self.projection = None
        
        device_name = "GPU" if device.type == "cuda" else "CPU"
        print(f"✅ Loaded SentenceTransformer: {model_name} (dim={self.source_dim}) on {device_name}")
    
    def forward(self, text):
        """Embed text into a fixed-size vector."""
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)
            embedding = embedding.clone().to(self.device)
        
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
            embeddings = embeddings.clone().to(self.device)
        
        if self.projection is not None:
            embeddings = self.projection(embeddings.float())
        
        return embeddings

