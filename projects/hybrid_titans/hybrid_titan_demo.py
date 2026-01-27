"""
Hybrid Titan Demo: TitanRAG Retrieval + FLAN-T5 Generation

This experiment demonstrates Direction A from the research plan:
- TitanRAG: Performs semantic retrieval using Memory + Ensemble Fusion
- Flan-T5-2: Generates natural language answers based on retrieved context

Architecture:
    Question → TitanRAG → Context → FLAN-T5-Large → Answer
"""

import torch
import torch.nn as nn
import argparse
import sys
import os
import re  # For sentence splitting

# Add parent directory to path (hybrid_titans -> projects -> TitanLLM)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

from main import TitanRAG
from data.sample_essays import ESSAY_CLIMATE, ESSAY_AI, ESSAY_SPACE, TEST_QUESTIONS


def split_into_chunks(text, min_length=30, sentences_per_chunk=2):
    """
    Split text into semantic chunks for retrieval.
    
    Strategy:
    - Headers are grouped with their following 2-3 sentences
    - Regular paragraphs are split into overlapping chunks
    - Maintains context continuity while being granular enough for precise retrieval
    
    Args:
        text: Input text (can contain markdown headers)
        min_length: Minimum chunk length to keep
        sentences_per_chunk: Number of sentences per regular chunk
        
    Returns:
        List of text chunks
    """
    lines = text.split('\n')
    chunks = []
    current_header = ""
    current_content = []
    
    def flush_content():
        """Flush accumulated content with header into chunks."""
        nonlocal current_header, current_content
        
        if not current_content:
            if current_header and len(current_header) > 5:
                chunks.append(current_header)
            current_header = ""
            return
            
        # Combine header with content
        content_text = " ".join(current_content)
        if current_header:
            full_chunk = f"{current_header} {content_text}"
        else:
            full_chunk = content_text
            
        if len(full_chunk) >= min_length:
            chunks.append(full_chunk)
        
        current_header = ""
        current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if it's a header
        if line.startswith('#'):
            # Flush previous section
            flush_content()
            current_header = line
            continue
        
        # Handle regular content - split into sentences
        # Protect abbreviations
        protected = line.replace('Dr.', 'Dr§').replace('Mr.', 'Mr§').replace('Ms.', 'Ms§')
        protected = protected.replace('etc.', 'etc§').replace('e.g.', 'eg§').replace('i.e.', 'ie§')
        
        # Split on sentence boundaries
        parts = re.split(r'(?<=[.!?])\s+', protected)
        
        for part in parts:
            part = part.strip()
            # Restore abbreviations
            part = part.replace('Dr§', 'Dr.').replace('Mr§', 'Mr.').replace('Ms§', 'Ms.')
            part = part.replace('etc§', 'etc.').replace('eg§', 'e.g.').replace('ie§', 'i.e.')
            
            if len(part) >= 10:  # Minimum sentence length
                current_content.append(part)
                
                # Create chunk when we have enough sentences
                if len(current_content) >= sentences_per_chunk:
                    flush_content()
    
    # Flush remaining content
    flush_content()
    
    return chunks


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
        with torch.no_grad():
            embedding = self.model.encode(text, convert_to_tensor=True)
            embedding = embedding.clone().cpu()
        
        if self.projection is not None:
            with torch.no_grad():
                embedding = self.projection(embedding)
        
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return embedding.unsqueeze(0)


class TrainableDecider(nn.Module):
    """
    Trainable Decision Maker using MLP classifier.
    
    Learns to predict when to use hybrid mode vs titans_only mode
    based on features extracted from the question and memory state.
    
    Features (5-dim):
    1. question_type_score: Factual (1.0) vs Reasoning (0.0)
    2. memory_confidence: LTM output strength (inverted: high confidence = low score)
    3. retrieval_dispersion: Score distribution entropy
    4. question_length: Normalized question length
    5. keyword_density: Factual keyword density
    
    Output: P(use_hybrid) ∈ [0, 1]
    """
    
    FACTUAL_KEYWORDS = ["what", "when", "where", "how many", "how much", "which", "who"]
    REASONING_KEYWORDS = ["why", "how does", "explain", "describe", "compare", "analyze"]
    
    def __init__(self, input_dim=5, hidden_dim=32, dropout=0.2, lr=0.01):  # Increased lr from 0.001 to 0.01
        super().__init__()
        
        # Build classifier with explicit layers for bias initialization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize final layer bias to 0 so sigmoid starts at 0.5
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        # Training components
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        
        # Training data buffer
        self.training_buffer = []
        self.buffer_max_size = 100
        
        # Statistics
        self.total_decisions = 0
        self.hybrid_decisions = 0
        
        # Dynamic threshold (learned from training data)
        self.learned_threshold = 0.5
        
    def extract_features(self, question, query_emb, titan_ltm, retrieval_scores=None):
        """
        Extract 5-dimensional feature vector from question and memory state.
        
        Returns:
            features: Tensor of shape (5,)
            feature_dict: Dict with named features for debugging
        """
        # Feature 1: Question type score (factual=1.0, reasoning=0.0)
        q_lower = question.lower()
        factual_count = sum(1 for kw in self.FACTUAL_KEYWORDS if kw in q_lower)
        reasoning_count = sum(1 for kw in self.REASONING_KEYWORDS if kw in q_lower)
        
        if factual_count > 0 and reasoning_count == 0:
            question_type_score = 1.0
        elif reasoning_count > 0 and factual_count == 0:
            question_type_score = 0.0
        elif factual_count > reasoning_count:
            question_type_score = 0.7
        elif reasoning_count > factual_count:
            question_type_score = 0.3
        else:
            question_type_score = 0.5
        
        # Feature 2: Memory confidence (inverted: high confidence = low need for RAG)
        with torch.no_grad():
            memory_output = titan_ltm.forward_no_update(query_emb)
            output_norm = memory_output.norm()
            input_norm = query_emb.norm()
            confidence_ratio = output_norm / (input_norm + 1e-8)
            # Invert: high confidence → 0.0 (titans only), low confidence → 1.0 (need RAG)
            memory_confidence = 1.0 - torch.clamp(confidence_ratio / 2.0, 0.0, 1.0).item()
        
        # Feature 3: Retrieval dispersion
        if retrieval_scores is not None and len(retrieval_scores) > 0:
            scores_tensor = torch.tensor(retrieval_scores) if not isinstance(retrieval_scores, torch.Tensor) else retrieval_scores
            if scores_tensor.max() - scores_tensor.min() < 1e-6:
                dispersion = 1.0
            else:
                normalized = (scores_tensor - scores_tensor.min()) / (scores_tensor.max() - scores_tensor.min() + 1e-8)
                normalized = normalized + 1e-8
                normalized = normalized / normalized.sum()
                entropy = -(normalized * torch.log(normalized)).sum().item()
                max_entropy = torch.log(torch.tensor(len(retrieval_scores), dtype=torch.float32)).item()
                dispersion = entropy / (max_entropy + 1e-8)
        else:
            dispersion = 0.5
        
        # Feature 4: Question length (normalized, assume max ~50 words)
        word_count = len(question.split())
        question_length = min(word_count / 50.0, 1.0)
        
        # Feature 5: Keyword density (factual keywords / total words)
        keyword_density = factual_count / max(word_count, 1)
        
        features = torch.tensor([
            question_type_score,
            memory_confidence,
            dispersion,
            question_length,
            keyword_density
        ], dtype=torch.float32)
        
        feature_dict = {
            "question_type": question_type_score,
            "memory_confidence": memory_confidence,
            "retrieval_dispersion": dispersion,
            "question_length": question_length,
            "keyword_density": keyword_density
        }
        
        return features, feature_dict
    
    def forward(self, features):
        """Forward pass: features → P(use_hybrid)"""
        x = torch.relu(self.fc1(features))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.sigmoid(x)
    
    def decide(self, question, query_emb, titan_ltm, retrieval_scores=None, threshold=None):
        """
        Make decision based on learned classifier.
        Uses learned_threshold if threshold is None.
        
        Returns:
            decision: "hybrid" or "titans_only"
            signals: Dict with decision details
        """
        features, feature_dict = self.extract_features(
            question, query_emb, titan_ltm, retrieval_scores
        )
        
        with torch.no_grad():
            prob_hybrid = self.forward(features).item()
        
        # Use learned threshold if not specified
        actual_threshold = threshold if threshold is not None else self.learned_threshold
        decision = "hybrid" if prob_hybrid > actual_threshold else "titans_only"
        
        # Update statistics
        self.total_decisions += 1
        if decision == "hybrid":
            self.hybrid_decisions += 1
        
        signals = {
            **feature_dict,
            "prob_hybrid": prob_hybrid,
            "threshold": actual_threshold,
            "final_score": prob_hybrid  # For compatibility with HybridDecider
        }
        
        return decision, signals
    
    def collect_sample(self, features, label):
        """
        Collect training sample.
        
        Args:
            features: 5-dim feature tensor
            label: 1.0 if hybrid was better, 0.0 if titans_only was better
        """
        self.training_buffer.append((features.clone(), label))
        
        # Keep buffer size limited
        if len(self.training_buffer) > self.buffer_max_size:
            self.training_buffer.pop(0)
    
    def train_step(self, batch_size=16):
        """
        Perform one training step on buffered samples.
        
        Returns:
            loss: Training loss, or None if not enough samples
        """
        if len(self.training_buffer) < batch_size:
            return None
        
        # Sample batch
        import random
        batch = random.sample(self.training_buffer, batch_size)
        
        features = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch], dtype=torch.float32).unsqueeze(1)
        
        # Forward pass
        self.train()
        self.optimizer.zero_grad()
        predictions = self.forward(features)
        loss = self.criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        self.eval()
        return loss.item()
    
    def train_on_buffer(self, epochs=10, batch_size=16):
        """
        Train on all buffered samples for multiple epochs.
        Uses class-balanced weighting for imbalanced data.
        
        Returns:
            losses: List of losses per epoch
        """
        if len(self.training_buffer) < 2:
            print(f"Not enough samples ({len(self.training_buffer)}/2 minimum)")
            return []
        
        # Calculate class weights for balanced training
        labels = [x[1] for x in self.training_buffer]
        n_hybrid = sum(labels)
        n_titans = len(labels) - n_hybrid
        
        if n_hybrid > 0 and n_titans > 0:
            # Inverse frequency weighting
            weight_hybrid = len(labels) / (2 * n_hybrid)
            weight_titans = len(labels) / (2 * n_titans)
            print(f"   Class balance: titans_only={n_titans}, hybrid={n_hybrid}")
            print(f"   Weights: titans={weight_titans:.2f}, hybrid={weight_hybrid:.2f}")
        else:
            weight_hybrid = 1.0
            weight_titans = 1.0
            print(f"   ⚠️ Imbalanced: all samples are {'hybrid' if n_hybrid > 0 else 'titans_only'}")
        
        # Prepare all data
        features = torch.stack([x[0] for x in self.training_buffer])
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        # Create sample weights
        sample_weights = torch.tensor([
            weight_hybrid if l == 1.0 else weight_titans 
            for l in labels
        ], dtype=torch.float32)
        
        actual_batch_size = min(batch_size, len(self.training_buffer))
        losses = []
        
        self.train()
        for epoch in range(epochs):
            # Shuffle indices
            indices = torch.randperm(len(self.training_buffer))[:actual_batch_size]
            
            batch_features = features[indices]
            batch_labels = labels_tensor[indices]
            batch_weights = sample_weights[indices]
            
            self.optimizer.zero_grad()
            predictions = self.forward(batch_features)
            
            # Weighted BCE loss
            bce = -batch_labels * torch.log(predictions + 1e-8) - (1 - batch_labels) * torch.log(1 - predictions + 1e-8)
            weighted_loss = (bce * batch_weights.unsqueeze(1)).mean()
            
            weighted_loss.backward()
            self.optimizer.step()
            losses.append(weighted_loss.item())
        
        self.eval()
        
        # Compute dynamic threshold based on predictions
        with torch.no_grad():
            final_preds = self.forward(features)
            pred_min = final_preds.min().item()
            pred_max = final_preds.max().item()
            pred_median = final_preds.median().item()
            
            # Set learned threshold to median of predictions
            # This ensures roughly 50% decisions each way
            self.learned_threshold = pred_median
            
            print(f"   Final prediction range: [{pred_min:.3f}, {pred_max:.3f}]")
            print(f"   Learned threshold (median): {self.learned_threshold:.3f}")
            n_above = (final_preds > self.learned_threshold).sum().item()
            print(f"   Predictions > threshold: {n_above}/{len(final_preds)}")
        
        return losses
    
    def save(self, path):
        """Save model and training buffer."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'training_buffer': self.training_buffer,
            'total_decisions': self.total_decisions,
            'hybrid_decisions': self.hybrid_decisions,
            'learned_threshold': self.learned_threshold
        }, path)
        print(f"✅ Saved TrainableDecider to {path}")
        print(f"   Learned threshold: {self.learned_threshold:.3f}")
    
    def load(self, path):
        """Load model and training buffer."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_buffer = checkpoint.get('training_buffer', [])
        self.total_decisions = checkpoint.get('total_decisions', 0)
        self.hybrid_decisions = checkpoint.get('hybrid_decisions', 0)
        self.learned_threshold = checkpoint.get('learned_threshold', 0.5)
        print(f"✅ Loaded TrainableDecider from {path}")
        print(f"   Buffer: {len(self.training_buffer)} samples, Decisions: {self.total_decisions}")
        print(f"   Learned threshold: {self.learned_threshold:.3f}")
    
    def get_stats(self):
        """Get decision statistics."""
        hybrid_ratio = self.hybrid_decisions / max(self.total_decisions, 1)
        return {
            "total_decisions": self.total_decisions,
            "hybrid_decisions": self.hybrid_decisions,
            "titans_only_decisions": self.total_decisions - self.hybrid_decisions,
            "hybrid_ratio": hybrid_ratio,
            "buffer_size": len(self.training_buffer)
        }


class HybridDecider:
    """
    Strategy 3: Hybrid Decision Maker
    
    Combines multiple signals to decide between:
    - titans_only: Use Titan memory directly (for conceptual/reasoning questions)
    - hybrid: Use RAG retrieval + Titan (for factual/precise questions)
    
    Signals:
    1. Question Type: Factual questions need RAG, reasoning questions use memory
    2. Memory Confidence: Strong memory signals can skip RAG
    3. Retrieval Dispersion: Concentrated retrieval scores indicate clear answer location
    """
    
    # Question type keywords
    FACTUAL_KEYWORDS = ["what", "when", "where", "how many", "how much", "which", "who"]
    REASONING_KEYWORDS = ["why", "how does", "explain", "describe", "compare", "analyze"]
    
    def __init__(self, 
                 question_weight=0.3, 
                 memory_weight=0.4, 
                 dispersion_weight=0.3,
                 hybrid_threshold=0.5):
        """
        Args:
            question_weight: Weight for question type signal
            memory_weight: Weight for memory confidence signal
            dispersion_weight: Weight for retrieval dispersion signal
            hybrid_threshold: If final score > threshold, use hybrid mode
        """
        self.question_weight = question_weight
        self.memory_weight = memory_weight
        self.dispersion_weight = dispersion_weight
        self.hybrid_threshold = hybrid_threshold
    
    def _question_type_score(self, question):
        """
        Analyze question type.
        Returns: 0.0-1.0, where 1.0 = needs RAG (factual), 0.0 = can use memory (reasoning)
        """
        q_lower = question.lower()
        
        # Check for factual keywords
        factual_count = sum(1 for kw in self.FACTUAL_KEYWORDS if kw in q_lower)
        
        # Check for reasoning keywords
        reasoning_count = sum(1 for kw in self.REASONING_KEYWORDS if kw in q_lower)
        
        # Normalize to 0-1
        if factual_count > 0 and reasoning_count == 0:
            return 1.0  # Clearly factual
        elif reasoning_count > 0 and factual_count == 0:
            return 0.0  # Clearly reasoning
        elif factual_count > reasoning_count:
            return 0.7  # Mostly factual
        elif reasoning_count > factual_count:
            return 0.3  # Mostly reasoning
        else:
            return 0.5  # Neutral/mixed
    
    def _memory_confidence(self, query_emb, titan_ltm):
        """
        Measure memory confidence based on LTM output strength.
        Returns: 0.0-1.0, where 1.0 = strong memory (can skip RAG), 0.0 = weak memory (need RAG)
        """
        with torch.no_grad():
            memory_output = titan_ltm.forward_no_update(query_emb)
            
            # Compute confidence as normalized output magnitude
            output_norm = memory_output.norm()
            input_norm = query_emb.norm()
            
            # Ratio > 1.0 means memory amplifies the signal (strong memory)
            # Ratio < 1.0 means memory weakens the signal (weak memory)
            confidence_ratio = output_norm / (input_norm + 1e-8)
            
            # Normalize to 0-1, where high confidence = can skip RAG (low score)
            # Invert: high confidence → 0.0 (titans only), low confidence → 1.0 (need RAG)
            return 1.0 - torch.clamp(confidence_ratio / 2.0, 0.0, 1.0).item()
    
    def _retrieval_dispersion(self, retrieval_scores):
        """
        Measure dispersion of retrieval scores.
        Returns: 0.0-1.0, where 1.0 = dispersed (need RAG), 0.0 = concentrated (memory enough)
        """
        if len(retrieval_scores) == 0:
            return 1.0
        
        # Normalize scores
        scores_tensor = torch.tensor(retrieval_scores) if not isinstance(retrieval_scores, torch.Tensor) else retrieval_scores
        
        if scores_tensor.max() - scores_tensor.min() < 1e-6:
            return 1.0  # All scores equal = very dispersed
        
        # Compute entropy-like dispersion
        normalized = (scores_tensor - scores_tensor.min()) / (scores_tensor.max() - scores_tensor.min() + 1e-8)
        normalized = normalized + 1e-8  # Avoid log(0)
        normalized = normalized / normalized.sum()  # Probability distribution
        
        # Entropy: high entropy = dispersed, low entropy = concentrated
        entropy = -(normalized * torch.log(normalized)).sum().item()
        max_entropy = torch.log(torch.tensor(len(retrieval_scores), dtype=torch.float32)).item()
        
        # Normalize to 0-1
        dispersion = entropy / (max_entropy + 1e-8)
        
        return dispersion
    
    def decide(self, question, query_emb, titan_ltm, retrieval_scores=None):
        """
        Make decision based on all signals.
        
        Args:
            question: Question text
            query_emb: Query embedding tensor
            titan_ltm: TitanMAG LTM module
            retrieval_scores: Optional pre-computed retrieval scores
        
        Returns:
            decision: "hybrid" or "titans_only"
            signals: Dict with individual signal scores for debugging
        """
        # Compute all signals
        question_score = self._question_type_score(question)
        memory_score = self._memory_confidence(query_emb, titan_ltm)
        
        if retrieval_scores is not None:
            dispersion_score = self._retrieval_dispersion(retrieval_scores)
        else:
            dispersion_score = 0.5  # Neutral if not available
        
        # Weighted combination
        final_score = (
            self.question_weight * question_score +
            self.memory_weight * memory_score +
            self.dispersion_weight * dispersion_score
        )
        
        # Decision
        decision = "hybrid" if final_score > self.hybrid_threshold else "titans_only"
        
        # Return decision and signals for transparency
        signals = {
            "question_type": question_score,
            "memory_confidence": memory_score,
            "retrieval_dispersion": dispersion_score,
            "final_score": final_score,
            "threshold": self.hybrid_threshold
        }
        
        return decision, signals


class HybridTitanRAG(nn.Module):
    """
    Hybrid Titan: TitanRAG Retrieval + LLM Generation
    
    This combines:
    1. TitanRAG for semantic memory-based retrieval
    2. External LLM (Flan-T5-2) for answer generation
    """
    
    def __init__(self, titan_rag, embedder, llm_name="google/flan-t5-large",
                 decision_threshold=0.5, question_weight=0.3, memory_weight=0.4, dispersion_weight=0.3,
                 fallback_confidence_threshold=0.3, enable_fallback=True,
                 use_trainable_decider=False, trainable_decider_lr=0.001, decider_model_path=None):
        super().__init__()
        self.titan_rag = titan_rag
        self.embedder = embedder
        
        # Load Flan-T5 for generation (better instruction following than Flan-T5-2)
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(llm_name)
        self.llm = T5ForConditionalGeneration.from_pretrained(llm_name)
        
        # Initialize Decision Maker (rule-based or trainable)
        self.use_trainable_decider = use_trainable_decider
        self.decision_threshold = decision_threshold
        
        if use_trainable_decider:
            self.decider = TrainableDecider(lr=trainable_decider_lr)
            if decider_model_path and os.path.exists(decider_model_path):
                self.decider.load(decider_model_path)
            print(f"✅ Initialized TrainableDecider (lr={trainable_decider_lr})")
        else:
            self.decider = HybridDecider(
                question_weight=question_weight,
                memory_weight=memory_weight,
                dispersion_weight=dispersion_weight,
                hybrid_threshold=decision_threshold
            )
            print(f"✅ Initialized HybridDecider (threshold={decision_threshold})")
        
        # Fallback settings: if titans_only confidence is low, fallback to hybrid
        self.fallback_confidence_threshold = fallback_confidence_threshold
        self.enable_fallback = enable_fallback
        
        # Decider model path for saving
        self.decider_model_path = decider_model_path
        
        print(f"✅ Loaded LLM: {llm_name}")
        if enable_fallback:
            print(f"✅ Fallback enabled (confidence < {fallback_confidence_threshold} → switch to hybrid)")
    
    def retrieve(self, question, essay_lines, line_embeddings, topk=7):
        """
        Retrieve relevant context using Ensemble Fusion.
        (Combines Keyword + Memory + Embedding for robust retrieval)
        
        Returns:
            context: Retrieved text
            details: Dict with retrieval scores
        """
        dim = self.embedder.target_dim
        
        # Embed question
        query_emb = self.embedder(question)
        query_flat = query_emb.reshape(-1, dim)
        
        with torch.no_grad():
            # Method 1: Keyword scores
            question_lower = question.lower()
            keyword_scores = []
            for line in essay_lines:
                score = sum(1 for word in question_lower.split() 
                           if len(word) > 3 and word in line.lower())
                keyword_scores.append(float(score))
            keyword_scores = torch.tensor(keyword_scores)
            
            # Method 2: Memory scores (LTM semantic)
            memory_output = self.titan_rag.titan.ltm.forward_no_update(query_flat)
            memory_vec = memory_output.mean(dim=0)
            memory_scores = torch.nn.functional.cosine_similarity(
                memory_vec.unsqueeze(0), line_embeddings
            )
            
            # Method 3: Direct embedding similarity
            query_vec = query_flat.mean(dim=0)
            embedding_scores = torch.nn.functional.cosine_similarity(
                query_vec.unsqueeze(0), line_embeddings
            )
            
            # Normalize
            def normalize(scores):
                min_s, max_s = scores.min(), scores.max()
                if max_s - min_s < 1e-6:
                    return torch.zeros_like(scores)
                return (scores - min_s) / (max_s - min_s)
            
            # Ensemble Fusion (balanced weights)
            fused_scores = (
                0.2 * normalize(keyword_scores) +
                0.5 * normalize(memory_scores) +
                0.3 * normalize(embedding_scores)
            )
            
            # Get top-k indices
            topk_values, topk_indices = fused_scores.topk(min(topk, len(essay_lines)))
            
            # Build context from top-k lines
            context_lines = [essay_lines[idx] for idx in topk_indices.tolist()]
            context = " ".join(context_lines)
        
        return context, {
            "keyword_top1": keyword_scores.argmax().item(),
            "memory_top1": memory_scores.argmax().item(),
            "embed_top1": embedding_scores.argmax().item(),
            "fused_top1": fused_scores.argmax().item(),
            "topk_indices": topk_indices.tolist()
        }
    
    def generate(self, question, context, max_new_tokens=100):
        """
        Generate answer using Flan-T5 with retrieved context.
        """
        # Improved prompt - encourages best-effort answer instead of giving up
        prompt = f"""Answer the question based on the context provided.
Extract specific details: exact numbers, names, dates, and technical terms.
Provide the most relevant answer from the context, even if partial.

Context:
{context}

Question: {question}

Answer:"""
        
        # Tokenize with increased max_length for more context
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=768  # Increased from 512 for more context
        )
        
        # Generate with beam search for better quality
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,  # Increased from 3 for better quality
                early_stopping=True,
                no_repeat_ngram_size=2  # Avoid repetition
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer, prompt
    
    def generate_from_memory(self, question, query_emb, essay_lines, line_embeddings, max_new_tokens=100, topk=4):
        """
        Generate answer using Titan memory-guided retrieval (titans_only mode).
        
        Key improvement: Uses Titan LTM output to find relevant context,
        rather than letting the LLM guess blindly.
        
        Strategy:
        1. Get memory-enhanced representation from Titan LTM
        2. Use memory output to find most similar sentences (memory-guided retrieval)
        3. Provide context to LLM for answer generation
        
        This achieves the "titans_only" goal of relying on Titan's learned memory
        while still providing factual grounding.
        """
        with torch.no_grad():
            # Step 1: Get memory-enhanced representation from Titan LTM
            memory_output = self.titan_rag.titan.ltm.forward_no_update(query_emb)
            memory_vec = memory_output.mean(dim=0)
            
            # Step 2: Memory-guided retrieval - use memory output to find relevant context
            # The memory has learned associations from digestion, so it "recalls" relevant info
            memory_scores = torch.nn.functional.cosine_similarity(
                memory_vec.unsqueeze(0), line_embeddings
            )
            
            # Get top-k sentences based on memory similarity (increased from 2 to 4)
            topk_values, topk_indices = memory_scores.topk(min(topk, len(essay_lines)))
            
            # Build context from memory-recalled sentences
            context_lines = [essay_lines[idx] for idx in topk_indices.tolist()]
            memory_context = " ".join(context_lines)
        
        # Step 3: Simplified prompt for memory-based recall
        prompt = f"""Answer the question based on the context from memory.
Extract specific details: exact numbers, names, dates, and technical terms.

Memory context:
{memory_context}

Question: {question}

Answer:"""
        
        # Tokenize with increased max_length
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=768  # Increased for more context
        )
        
        # Generate with improved settings
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,  # Increased from 3
                early_stopping=True,
                no_repeat_ngram_size=2  # Avoid repetition
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer, prompt, {
            "memory_topk_indices": topk_indices.tolist(),
            "memory_context": memory_context[:200] + "..." if len(memory_context) > 200 else memory_context
        }
    
    def answer(self, question, essay_lines, line_embeddings, expected_answer=None, topk=5, max_new_tokens=100):
        """
        Full pipeline with Decision Maker: 
        1. Decide whether to use titans_only or hybrid mode
        2. Generate answer accordingly
        3. Learn from Q&A (Online Learning)
        
        Args:
            question: The question to answer
            essay_lines: List of essay lines
            line_embeddings: Tensor of line embeddings
            expected_answer: If provided, model learns from this Q&A pair (Online Learning)
            topk: Number of context lines to retrieve (for hybrid mode)
            max_new_tokens: Max tokens to generate
        """
        dim = self.embedder.target_dim
        
        # Embed question
        query_emb = self.embedder(question)
        query_flat = query_emb.reshape(-1, dim)
        
        # Step 0: DECISION - Do we need RAG or can we use memory directly?
        # First do a quick retrieval to get dispersion signal
        with torch.no_grad():
            # Quick keyword scores for dispersion
            question_lower = question.lower()
            keyword_scores = torch.tensor([
                sum(1 for word in question_lower.split() 
                   if len(word) > 3 and word in line.lower())
                for line in essay_lines
            ], dtype=torch.float32)
            
            # Memory scores for dispersion
            memory_output = self.titan_rag.titan.ltm.forward_no_update(query_flat)
            memory_vec = memory_output.mean(dim=0)
            memory_scores = torch.nn.functional.cosine_similarity(
                memory_vec.unsqueeze(0), line_embeddings
            )
            
            # Combined scores for dispersion analysis
            combined_scores = keyword_scores + memory_scores
        
        # Make decision
        decision, decision_signals = self.decider.decide(
            question, 
            query_flat, 
            self.titan_rag.titan.ltm,
            retrieval_scores=combined_scores
        )
        
        # Step 1 & 2: Generate based on decision (with fallback mechanism)
        fallback_triggered = False
        
        if decision == "titans_only":
            # Try Titan memory-guided retrieval first
            answer, prompt, memory_details = self.generate_from_memory(
                question, query_flat, essay_lines, line_embeddings, max_new_tokens, topk=2
            )
            context = f"[Memory-Guided Retrieval] {memory_details['memory_context']}"
            retrieval_details = {
                "mode": "titans_only",
                "memory_indices": memory_details["memory_topk_indices"]
            }
            
            # FALLBACK CHECK: If answer is too short or memory confidence is low, try hybrid
            if self.enable_fallback:
                # Check 1: Answer too short (likely low quality)
                answer_too_short = len(answer.split()) < 3
                
                # Check 2: Memory confidence was low (from decision signals)
                memory_confidence_low = decision_signals["memory_confidence"] > (1 - self.fallback_confidence_threshold)
                
                # Check 3: Answer looks like a fallback pattern (e.g., just repeating question)
                question_words = set(question.lower().split())
                answer_words = set(answer.lower().split())
                high_overlap = len(question_words & answer_words) / max(len(answer_words), 1) > 0.7
                
                should_fallback = answer_too_short or (memory_confidence_low and high_overlap)
                
                if should_fallback:
                    # FALLBACK: Switch to hybrid mode
                    fallback_triggered = True
                    context, retrieval_details = self.retrieve(
                        question, essay_lines, line_embeddings, topk
                    )
                    answer, prompt = self.generate(question, context, max_new_tokens)
                    retrieval_details["mode"] = "hybrid_fallback"
                    retrieval_details["fallback_reason"] = "low_confidence" if memory_confidence_low else "short_answer"
        else:
            # Use full hybrid: RAG retrieval + generation
            context, retrieval_details = self.retrieve(
                question, essay_lines, line_embeddings, topk
            )
            answer, prompt = self.generate(question, context, max_new_tokens)
            retrieval_details["mode"] = "hybrid"
        
        # Step 3: Online Learning (True TTT!) - if expected answer provided
        learned = False
        if expected_answer is not None:
            expected_emb = self.embedder(expected_answer)
            expected_vec = expected_emb.reshape(-1, dim).mean(dim=0)
            
            # Update memory: Question → Expected Answer
            self.titan_rag.titan.ltm.forward_with_update(
                query_flat.mean(dim=0, keepdim=True),  # key = question
                expected_vec.unsqueeze(0)              # value = answer
            )
            learned = True
        
        return {
            "answer": answer,
            "context": context,
            "retrieval": retrieval_details,
            "decision": decision,
            "decision_signals": decision_signals,
            "prompt": prompt,
            "online_learned": learned,
            "fallback_triggered": fallback_triggered,
            "query_flat": query_flat,  # For training data collection
            "combined_scores": combined_scores  # For training data collection
        }
    
    def evaluate_answer_quality(self, answer, expected_answer):
        """
        Evaluate answer quality by comparing to expected answer.
        
        Returns:
            score: Quality score in [0, 1]
        """
        if not expected_answer:
            return 0.5
        
        # Keyword match score
        expected_words = set(word.lower() for word in expected_answer.split() if len(word) > 3)
        answer_words = set(word.lower() for word in answer.split() if len(word) > 3)
        
        if not expected_words:
            return 0.5
        
        overlap = len(expected_words & answer_words)
        keyword_score = overlap / len(expected_words)
        
        # Length penalty (too short = bad)
        length_penalty = 1.0 if len(answer.split()) >= 3 else 0.5
        
        return keyword_score * length_penalty
    
    def answer_with_training(self, question, essay_lines, line_embeddings, expected_answer=None, 
                             topk=5, max_new_tokens=100):
        """
        Answer question and collect training data for the TrainableDecider.
        
        This method:
        1. Tries BOTH modes (titans_only and hybrid)
        2. Compares answer quality
        3. Collects training sample with the better mode as label
        
        Args:
            question: The question to answer
            essay_lines: List of essay lines
            line_embeddings: Tensor of line embeddings
            expected_answer: If provided, used for quality comparison
            topk: Number of context lines for hybrid mode
            max_new_tokens: Max tokens to generate
            
        Returns:
            result: Dict with answers from both modes and training info
        """
        if not self.use_trainable_decider:
            # Fall back to regular answer for rule-based decider
            return self.answer(question, essay_lines, line_embeddings, 
                             expected_answer, topk, max_new_tokens)
        
        dim = self.embedder.target_dim
        
        # Embed question
        query_emb = self.embedder(question)
        query_flat = query_emb.reshape(-1, dim)
        
        # Get retrieval scores for feature extraction
        with torch.no_grad():
            question_lower = question.lower()
            keyword_scores = torch.tensor([
                sum(1 for word in question_lower.split() 
                   if len(word) > 3 and word in line.lower())
                for line in essay_lines
            ], dtype=torch.float32)
            
            memory_output = self.titan_rag.titan.ltm.forward_no_update(query_flat)
            memory_vec = memory_output.mean(dim=0)
            memory_scores = torch.nn.functional.cosine_similarity(
                memory_vec.unsqueeze(0), line_embeddings
            )
            combined_scores = keyword_scores + memory_scores
        
        # Extract features for training
        features, feature_dict = self.decider.extract_features(
            question, query_flat, self.titan_rag.titan.ltm, combined_scores
        )
        
        # Try titans_only mode
        titans_answer, titans_prompt, titans_details = self.generate_from_memory(
            question, query_flat, essay_lines, line_embeddings, max_new_tokens, topk=2
        )
        titans_quality = self.evaluate_answer_quality(titans_answer, expected_answer)
        
        # Try hybrid mode
        hybrid_context, hybrid_retrieval = self.retrieve(
            question, essay_lines, line_embeddings, topk
        )
        hybrid_answer, hybrid_prompt = self.generate(question, hybrid_context, max_new_tokens)
        hybrid_quality = self.evaluate_answer_quality(hybrid_answer, expected_answer)
        
        # Determine label: 1.0 if hybrid is better, 0.0 if titans_only is better
        # Intelligent tie-breaker based on question type:
        # - For reasoning questions (question_type_score < 0.5): prefer hybrid (more context helps)
        # - For factual questions (question_type_score >= 0.5): prefer titans_only (memory lookup)
        question_type_score = feature_dict["question_type"]
        
        if hybrid_quality > titans_quality + 0.05:
            label = 1.0  # Hybrid is clearly better
            best_mode = "hybrid"
            best_answer = hybrid_answer
        elif titans_quality > hybrid_quality + 0.05:
            label = 0.0  # Titans_only is clearly better
            best_mode = "titans_only"
            best_answer = titans_answer
        else:
            # Tie or very close: use question type as tie-breaker
            if question_type_score < 0.5:
                # Reasoning question: prefer hybrid (needs more context/reasoning)
                label = 1.0
                best_mode = "hybrid"
                best_answer = hybrid_answer
            else:
                # Factual question: prefer titans_only (direct memory recall)
                label = 0.0
                best_mode = "titans_only"
                best_answer = titans_answer
        
        # Collect training sample
        self.decider.collect_sample(features, label)
        
        # Get current decision for comparison
        decision, decision_signals = self.decider.decide(
            question, query_flat, self.titan_rag.titan.ltm, combined_scores
        )
        
        # Online learning for Titan memory
        learned = False
        if expected_answer is not None:
            expected_emb = self.embedder(expected_answer)
            expected_vec = expected_emb.reshape(-1, dim).mean(dim=0)
            self.titan_rag.titan.ltm.forward_with_update(
                query_flat.mean(dim=0, keepdim=True),
                expected_vec.unsqueeze(0)
            )
            learned = True
        
        return {
            "answer": best_answer,
            "best_mode": best_mode,
            "decision": decision,
            "decision_correct": (decision == best_mode),
            "decision_signals": decision_signals,
            "titans_only_answer": titans_answer,
            "titans_only_quality": titans_quality,
            "hybrid_answer": hybrid_answer,
            "hybrid_quality": hybrid_quality,
            "label": label,
            "features": feature_dict,
            "online_learned": learned,
            "buffer_size": len(self.decider.training_buffer)
        }
    
    def train_decider(self, epochs=10, batch_size=16):
        """
        Train the TrainableDecider on collected samples.
        
        Returns:
            losses: List of losses per epoch
        """
        if not self.use_trainable_decider:
            print("⚠️ TrainableDecider not enabled")
            return []
        
        losses = self.decider.train_on_buffer(epochs, batch_size)
        
        if losses:
            print(f"✅ Trained decider: {len(losses)} steps, final loss: {losses[-1]:.4f}")
            
            # Save model if path specified
            if self.decider_model_path:
                self.decider.save(self.decider_model_path)
        
        return losses
    
    def get_decider_stats(self):
        """Get decision maker statistics."""
        if self.use_trainable_decider:
            return self.decider.get_stats()
        else:
            return {"type": "HybridDecider (rule-based)"}


def run_hybrid_titan_demo(args):
    """Run the Hybrid Titan demo."""
    
    print("=" * 60)
    print("HYBRID TITAN DEMO")
    print("TitanRAG Retrieval + Flan-T5-2 Generation")
    print("=" * 60)
    
    # Select essay based on argument
    essay_key = args.essay.lower()
    if essay_key == "climate":
        essay_text = ESSAY_CLIMATE
        essay_questions = TEST_QUESTIONS["climate"]
    elif essay_key == "ai":
        essay_text = ESSAY_AI
        essay_questions = TEST_QUESTIONS["ai"]
    elif essay_key == "space":
        essay_text = ESSAY_SPACE
        essay_questions = TEST_QUESTIONS["space"]
    else:
        essay_text = ESSAY_CLIMATE
        essay_questions = TEST_QUESTIONS["climate"]
    
    print(f"\n📖 Selected Essay: {essay_key.upper()}")
    print(f"   Length: {len(essay_text)} characters")
    
    # Initialize embedder
    embedder = SentenceTransformerEmbedder(target_dim=args.dim)
    
    # Initialize TitanRAG (first create base model, then wrap)
    from main import TitanMAG
    
    base_model = TitanMAG(
        dim=args.dim, 
        window_size=32, 
        hidden_dim=args.dim * 2,
        memory_depth=3, 
        num_persistent_tokens=4, 
        threshold=0.0
    )
    
    # Configure for high-fidelity memory
    with torch.no_grad():
        base_model.ltm.theta.fill_(0.01)   # Learning rate
        base_model.ltm.alpha.fill_(0.0001) # Minimal forgetting
    
    titan_rag = TitanRAG(base_model)
    
    # Parse essay into semantic CHUNKS (headers + 2 sentences each)
    # This preserves context while enabling granular retrieval
    essay_chunks = split_into_chunks(essay_text, min_length=30, sentences_per_chunk=2)
    
    print(f"\n🔄 Embedding {len(essay_chunks)} chunks...")
    
    # Embed all chunks
    with torch.no_grad():
        chunk_embeddings = []
        for chunk in essay_chunks:
            chunk_emb = embedder(chunk)
            chunk_vec = chunk_emb.reshape(-1, args.dim).mean(dim=0)
            chunk_embeddings.append(chunk_vec)
        line_embeddings = torch.stack(chunk_embeddings)  # Keep variable name for compatibility
    
    # Use chunks for retrieval
    essay_lines = essay_chunks  # Alias for compatibility
    
    # Digest into TitanRAG
    print(f"\n🧠 Digesting into TitanRAG ({args.epochs} epochs)...")
    all_embeddings = torch.cat([embedder(line) for line in essay_lines], dim=1)
    
    for epoch in range(args.epochs):
        titan_rag.digest_knowledge(all_embeddings)
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{args.epochs} completed")
    
    # Initialize Hybrid Titan with Flan-T5 and Decision Maker
    print("\n🤖 Initializing Hybrid Titan (Flan-T5 generator + Decision Maker)...")
    
    # Determine decider model path
    decider_model_path = None
    if args.use_trainable_decider:
        decider_model_path = args.decider_model_path or f"decider_model_{essay_key}.pt"
    
    hybrid_titan = HybridTitanRAG(
        titan_rag, 
        embedder,
        decision_threshold=args.decision_threshold,
        question_weight=args.question_weight,
        memory_weight=args.memory_weight,
        dispersion_weight=args.dispersion_weight,
        fallback_confidence_threshold=args.fallback_threshold,
        enable_fallback=not args.disable_fallback,
        use_trainable_decider=args.use_trainable_decider,
        trainable_decider_lr=args.decider_lr,
        decider_model_path=decider_model_path
    )
    
    # Run Q&A with Decision Maker + Online Learning
    print("\n" + "=" * 60)
    if args.use_trainable_decider and args.train_decider:
        print("TRAINING MODE: Collecting data for TrainableDecider")
    else:
        print("QUESTION ANSWERING with DECISION MAKER + ONLINE LEARNING")
    print("=" * 60)
    print(f"Decision Threshold: {args.decision_threshold}")
    if args.use_trainable_decider:
        print(f"Decider: TrainableDecider (lr={args.decider_lr})")
    else:
        print(f"Decider: HybridDecider (Rule-based)")
        print(f"Weights - Question: {args.question_weight}, Memory: {args.memory_weight}, Dispersion: {args.dispersion_weight}")
    if not args.disable_fallback:
        print(f"Fallback Enabled: titans_only → hybrid when confidence < {args.fallback_threshold}")
    else:
        print("Fallback DISABLED: showing pure decider output")
    
    # Training mode tracking
    training_correct = 0
    training_total = 0
    
    for i, (question, expected) in enumerate(essay_questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Q{i}: {question}")
        print(f"Expected: {expected}")
        
        # Choose between training mode and inference mode
        if args.use_trainable_decider and args.train_decider:
            # TRAINING MODE: Try both modes and collect data
            result = hybrid_titan.answer_with_training(
                question, 
                essay_lines, 
                line_embeddings,
                expected_answer=expected,
                topk=args.topk,
                max_new_tokens=args.max_tokens
            )
            
            # Show training info
            print(f"\n[Training Mode]")
            print(f"  Titans-Only Quality: {result['titans_only_quality']:.2f}")
            print(f"  Hybrid Quality: {result['hybrid_quality']:.2f}")
            print(f"  Best Mode: {result['best_mode']} (label={result['label']})")
            print(f"  Current Decision: {result['decision']} ({'✅ Correct' if result['decision_correct'] else '❌ Wrong'})")
            print(f"  Buffer Size: {result['buffer_size']}")
            
            # Track accuracy
            if result['decision_correct']:
                training_correct += 1
            training_total += 1
            
            # Show signals
            signals = result['decision_signals']
            print(f"  Signals: Q-Type={signals['question_type']:.2f}, Memory={signals['memory_confidence']:.2f}, Dispersion={signals['retrieval_dispersion']:.2f}")
            
            print(f"\n[Best Answer ({result['best_mode']})]")
            print(f"  {result['answer']}")
            
        else:
            # INFERENCE MODE: Use decider to choose mode
            result = hybrid_titan.answer(
                question, 
                essay_lines, 
                line_embeddings,
                expected_answer=expected,  # Enable Online Learning
                topk=args.topk,
                max_new_tokens=args.max_tokens
            )

            # Show Decision Maker output
            decision = result['decision']
            signals = result['decision_signals']
            mode_emoji = "🧠" if decision == "titans_only" else "🔍"
            print(f"\n[Decision Maker] {mode_emoji} Mode: {decision.upper()}")
            print(f"  Signals: Q-Type={signals['question_type']:.2f}, Memory={signals['memory_confidence']:.2f}, Dispersion={signals['retrieval_dispersion']:.2f}")
            print(f"  Final Score: {signals['final_score']:.2f} (threshold={signals['threshold']})")

            # Show retrieval/context info
            mode = result['retrieval'].get('mode', decision)
            if mode == "hybrid_fallback":
                fallback_reason = result['retrieval'].get('fallback_reason', 'unknown')
                print(f"\n[🔄 FALLBACK: titans_only → hybrid] Reason: {fallback_reason}")
                print(f"[Hybrid Retrieval] Indices: {result['retrieval'].get('topk_indices', 'N/A')}")
            elif decision == "hybrid":
                print(f"\n[Hybrid Retrieval] Indices: {result['retrieval'].get('topk_indices', 'N/A')}")
            else:
                print(f"\n[Memory-Guided Retrieval] Indices: {result['retrieval'].get('memory_indices', 'N/A')}")
            print(f"[Context]\n  {result['context'][:200]}..." if len(result['context']) > 200 else f"[Context]\n  {result['context']}")

            print(f"\n[Generated Answer]")
            print(f"  {result['answer']}")

        # Show Online Learning status
        if result.get('online_learned'):
            print("  📚 Online Learning: Updated memory with this Q&A")
        
        # Simple evaluation: check if expected answer keywords appear
        expected_words = expected.lower().split()
        answer_lower = result['answer'].lower()
        matches = sum(1 for word in expected_words if len(word) > 3 and word in answer_lower)
        match_ratio = matches / len([w for w in expected_words if len(w) > 3]) if expected_words else 0
        
        if match_ratio > 0.3:
            print(f"  ✅ Contains expected keywords ({match_ratio:.0%})")
        else:
            print(f"  ⚠️ Missing expected keywords ({match_ratio:.0%})")
    
    # Post-loop training for TrainableDecider
    if args.use_trainable_decider and args.train_decider:
        print(f"\n{'─' * 60}")
        print(f"TRAINING SUMMARY")
        print(f"{'─' * 60}")
        print(f"Pre-training Decision Accuracy: {training_correct}/{training_total} ({training_correct/max(training_total,1)*100:.1f}%)")
        
        # Train the decider
        print(f"\nTraining TrainableDecider on {len(hybrid_titan.decider.training_buffer)} samples...")
        losses = hybrid_titan.train_decider(epochs=args.train_epochs, batch_size=min(8, len(hybrid_titan.decider.training_buffer)))
        
        if losses:
            print(f"Training complete! Final loss: {losses[-1]:.4f}")
            print(f"Model saved to: {hybrid_titan.decider_model_path}")
        
        # Show stats
        stats = hybrid_titan.get_decider_stats()
        print(f"\nDecider Stats:")
        print(f"  Total Decisions: {stats['total_decisions']}")
        print(f"  Hybrid Ratio: {stats['hybrid_ratio']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✅ Hybrid Titan Demo Complete!")
    print("=" * 60)
    
    return hybrid_titan


def main():
    parser = argparse.ArgumentParser(description="Hybrid Titan Demo with Decision Maker")
    parser.add_argument("--essay", type=str, default="climate",
                       choices=["climate", "ai", "space"],
                       help="Essay to use")
    parser.add_argument("--dim", type=int, default=256,
                       help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Digestion epochs (default: 100 for better memory)")
    parser.add_argument("--topk", type=int, default=3,
                       help="Top-k lines to retrieve")
    parser.add_argument("--max_tokens", type=int, default=50,
                       help="Max tokens to generate")
    
    # Decision Maker parameters
    parser.add_argument("--decision_threshold", type=float, default=0.5,
                       help="Decision threshold (>threshold = hybrid, <=threshold = titans_only)")
    parser.add_argument("--question_weight", type=float, default=0.3,
                       help="Weight for question type signal")
    parser.add_argument("--memory_weight", type=float, default=0.4,
                       help="Weight for memory confidence signal")
    parser.add_argument("--dispersion_weight", type=float, default=0.3,
                       help="Weight for retrieval dispersion signal")
   
    # Fallback parameters
    parser.add_argument("--disable_fallback", action="store_true",
                       help="Disable fallback from titans_only to hybrid (show pure decider output)")
    parser.add_argument("--fallback_threshold", type=float, default=0.3,
                       help="Fallback confidence threshold (lower = more fallbacks)")
    
    # Trainable Decider parameters
    parser.add_argument("--use_trainable_decider", action="store_true",
                       help="Use trainable MLP-based decision maker instead of rule-based")
    parser.add_argument("--train_decider", action="store_true",
                       help="Enable training mode: collect data and train the decider")
    parser.add_argument("--decider_lr", type=float, default=0.001,
                       help="Learning rate for TrainableDecider")
    parser.add_argument("--decider_model_path", type=str, default=None,
                       help="Path to save/load TrainableDecider model")
    parser.add_argument("--train_epochs", type=int, default=40,
                       help="Number of epochs to train the decider")
    
    args = parser.parse_args()
    run_hybrid_titan_demo(args)


if __name__ == "__main__":
    main()
