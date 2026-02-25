"""
Baseline Retrieval Strategies for TitanRAG evaluation.

Three modes:
- PureRAG: BM25 + Embedding (no Memory)
- TitanOnly: Memory-guided retrieval only
- HybridRAG: BM25 + Memory + Embedding fusion
- AdaptiveHybridRAG: Per-chunk soft routing (scorer-guided RAG/Titan blending)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """Normalize scores to [0, 1] range."""
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-6:
        return torch.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def compute_bm25_scores(question: str, documents: List[str]) -> torch.Tensor:
    """
    Compute BM25 scores with TF-IDF weighting.
    
    Uses proper BM25 formula with:
    - Term frequency with saturation (k1=1.5)
    - Inverse document frequency 
    - Document length normalization (b=0.75)
    """
    import re
    import math
    
    STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
                 'to', 'for', 'of', 'with', 'by', 'and', 'or', 'not', 'that',
                 'this', 'it', 'from', 'as', 'be', 'has', 'had', 'have', 'do',
                 'does', 'did', 'but', 'if', 'so', 'what', 'which', 'who',
                 'how', 'when', 'where', 'why', 'can', 'will', 'would', 'could'}
    
    def tokenize(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if len(w) > 1 and w not in STOPWORDS]
    
    query_terms = tokenize(question)
    if not query_terms:
        return torch.zeros(len(documents))
    
    # Tokenize all documents
    doc_tokens = [tokenize(doc) for doc in documents]
    avg_dl = sum(len(dt) for dt in doc_tokens) / max(len(doc_tokens), 1)
    
    # BM25 parameters
    k1, b = 1.5, 0.75
    N = len(documents)
    
    scores = []
    for doc_toks in doc_tokens:
        dl = len(doc_toks)
        score = 0.0
        tf_map = {}
        for t in doc_toks:
            tf_map[t] = tf_map.get(t, 0) + 1
        
        for term in query_terms:
            # Document frequency
            df = sum(1 for dt in doc_tokens if term in dt)
            if df == 0:
                continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            tf = tf_map.get(term, 0)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            score += idf * tf_norm
        
        scores.append(score)
    
    return torch.tensor(scores)


class BaseRetriever:
    """Base class for retrieval strategies."""
    
    def __init__(self, embedder, llm_generator, multihop=False):
        self.embedder = embedder
        self.llm = llm_generator
        self.multihop = multihop
    
    def retrieve(self, question: str, documents: List[str], 
                 doc_embeddings: torch.Tensor, topk: int = 5) -> Tuple[str, Dict]:
        """
        Retrieve relevant context from documents.
        
        Args:
            question: Query text
            documents: List of document strings
            doc_embeddings: Pre-computed document embeddings
            topk: Number of documents to retrieve
        
        Returns:
            tuple: (context_string, retrieval_details)
        """
        raise NotImplementedError
    
    def answer(self, question: str, documents: List[str],
               doc_embeddings: torch.Tensor, topk: int = 5,
               max_new_tokens: int = 100) -> Dict:
        """
        Full pipeline: retrieve + generate answer.
        
        Returns:
            dict: {"answer": str, "context": str, "retrieval_details": dict}
        """
        context, details = self.retrieve(question, documents, doc_embeddings, topk)
        answer, prompt = self.llm.generate_qa(question, context, max_new_tokens,
                                                multihop=self.multihop)
        
        return {
            "answer": answer,
            "context": context,
            "prompt": prompt,
            "retrieval_details": details,
            "mode": self.__class__.__name__
        }


class PureRAG(BaseRetriever):
    """
    Pure RAG: BM25 + Embedding retrieval (no Memory).
    
    This is the traditional RAG baseline without any Titan memory.
    """
    
    def __init__(self, embedder, llm_generator, 
                 bm25_weight: float = 0.4, embed_weight: float = 0.6,
                 multihop: bool = False):
        super().__init__(embedder, llm_generator, multihop=multihop)
        self.bm25_weight = bm25_weight
        self.embed_weight = embed_weight
    
    def retrieve(self, question: str, documents: List[str],
                 doc_embeddings: torch.Tensor, topk: int = 5) -> Tuple[str, Dict]:
        """Retrieve using BM25 + Embedding only."""
        
        # Embed question
        query_emb = self.embedder(question)
        query_vec = query_emb.reshape(-1, self.embedder.target_dim).mean(dim=0)
        
        with torch.no_grad():
            # BM25 scores
            bm25_scores = compute_bm25_scores(question, documents)
            
            # Embedding similarity
            embed_scores = F.cosine_similarity(
                query_vec.unsqueeze(0), doc_embeddings
            )
            
            # Fusion (no Memory!)
            # Move BM25 scores to same device as embeddings
            bm25_scores = bm25_scores.to(doc_embeddings.device)
            
            fused_scores = (
                self.bm25_weight * normalize_scores(bm25_scores) +
                self.embed_weight * normalize_scores(embed_scores)
            )
            
            # Top-K
            topk_values, topk_indices = fused_scores.topk(min(topk, len(documents)))
        
        # Build context
        context_lines = [documents[idx] for idx in topk_indices.tolist()]
        context = "\n\n".join(context_lines)
        
        return context, {
            "mode": "PureRAG",
            "topk_indices": topk_indices.tolist(),
            "bm25_top1": bm25_scores.argmax().item(),
            "embed_top1": embed_scores.argmax().item(),
            "fused_top1": fused_scores.argmax().item()
        }


class TitanOnly(BaseRetriever):
    """
    Titan-Only: Memory-guided retrieval.
    
    Uses only the Titan memory output for retrieval scoring.
    No BM25, no direct embedding similarity.
    """
    
    def __init__(self, embedder, llm_generator, titan_rag, multihop=False):
        super().__init__(embedder, llm_generator, multihop=multihop)
        self.titan_rag = titan_rag
    
    def retrieve(self, question: str, documents: List[str],
                 doc_embeddings: torch.Tensor, topk: int = 5) -> Tuple[str, Dict]:
        """Retrieve using Memory output only."""
        
        dim = self.embedder.target_dim
        
        # Embed question
        query_emb = self.embedder(question)
        query_flat = query_emb.reshape(-1, dim)
        
        with torch.no_grad():
            # Get memory output
            memory_output = self.titan_rag.titan.ltm.forward_no_update(query_flat)
            memory_vec = memory_output.mean(dim=0)
            
            # Memory-based similarity (only signal!)
            memory_scores = F.cosine_similarity(
                memory_vec.unsqueeze(0), doc_embeddings
            )
            
            # Top-K
            topk_values, topk_indices = memory_scores.topk(min(topk, len(documents)))
        
        # Build context
        context_lines = [documents[idx] for idx in topk_indices.tolist()]
        context = "\n\n".join(context_lines)
        
        return context, {
            "mode": "TitanOnly",
            "topk_indices": topk_indices.tolist(),
            "memory_top1": memory_scores.argmax().item()
        }
    
    def online_learn(self, question: str, expected_answer: str):
        """Learn from Q&A pair (Online Learning)."""
        dim = self.embedder.target_dim
        
        query_emb = self.embedder(question)
        query_vec = query_emb.reshape(-1, dim).mean(dim=0, keepdim=True)
        
        answer_emb = self.embedder(expected_answer)
        answer_vec = answer_emb.reshape(-1, dim).mean(dim=0, keepdim=True)
        
        self.titan_rag.titan.ltm.forward_with_update(query_vec, answer_vec)


class HybridRAG(BaseRetriever):
    """
    Hybrid RAG: BM25 + Memory + Embedding fusion with dynamic weighting.
    
    Key improvement: Memory weight is adjusted dynamically based on 
    memory confidence (output strength). Low confidence = reduce memory weight.
    """
    
    def __init__(self, embedder, llm_generator, titan_rag,
                 base_bm25_weight: float = 0.3,
                 base_memory_weight: float = 0.3,
                 base_embed_weight: float = 0.4,
                 multihop: bool = False):
        super().__init__(embedder, llm_generator, multihop=multihop)
        self.titan_rag = titan_rag
        self.base_bm25_weight = base_bm25_weight
        self.base_memory_weight = base_memory_weight
        self.base_embed_weight = base_embed_weight
    
    def _compute_memory_confidence(self, query_emb, memory_output):
        """
        Compute memory confidence based on output strength.
        
        Higher confidence when memory output norm is similar to input norm.
        Returns value in [0, 1] where 1 = high confidence.
        """
        with torch.no_grad():
            input_norm = torch.norm(query_emb)
            output_norm = torch.norm(memory_output)
            
            # Ratio of output to input norm
            confidence_ratio = output_norm / (input_norm + 1e-8)
            
            # Normalize to 0-1 range (assuming typical ratio is 0-2)
            confidence = torch.clamp(confidence_ratio / 2.0, 0.0, 1.0).item()
            
            return confidence
    
    def retrieve(self, question: str, documents: List[str],
                 doc_embeddings: torch.Tensor, topk: int = 5) -> Tuple[str, Dict]:
        """Retrieve using BM25 + Memory + Embedding fusion with dynamic weights."""
        
        dim = self.embedder.target_dim
        
        # Embed question
        query_emb = self.embedder(question)
        query_flat = query_emb.reshape(-1, dim)
        query_vec = query_flat.mean(dim=0)
        
        with torch.no_grad():
            # Signal 1: BM25
            bm25_scores = compute_bm25_scores(question, documents)
            
            # Signal 2: Memory
            memory_output = self.titan_rag.titan.ltm.forward_no_update(query_flat)
            memory_vec = memory_output.mean(dim=0)
            memory_scores = F.cosine_similarity(
                memory_vec.unsqueeze(0), doc_embeddings
            )
            
            # Signal 3: Direct Embedding
            embed_scores = F.cosine_similarity(
                query_vec.unsqueeze(0), doc_embeddings
            )
            
            # Compute memory confidence
            memory_confidence = self._compute_memory_confidence(query_flat, memory_output)
            
            # Dynamic weight adjustment based on confidence
            # Low confidence → reduce memory weight, increase embedding weight
            memory_weight = self.base_memory_weight * memory_confidence
            embed_weight = self.base_embed_weight + self.base_memory_weight * (1 - memory_confidence)
            bm25_weight = self.base_bm25_weight
            
            # Normalize weights to sum to 1
            total = bm25_weight + memory_weight + embed_weight
            bm25_weight /= total
            memory_weight /= total
            embed_weight /= total
            
            # Fusion with dynamic weights
            # Move BM25 scores to same device
            bm25_scores = bm25_scores.to(doc_embeddings.device)
            
            fused_scores = (
                bm25_weight * normalize_scores(bm25_scores) +
                memory_weight * normalize_scores(memory_scores) +
                embed_weight * normalize_scores(embed_scores)
            )
            
            # Top-K
            topk_values, topk_indices = fused_scores.topk(min(topk, len(documents)))
        
        # Build context
        context_lines = [documents[idx] for idx in topk_indices.tolist()]
        context = "\n\n".join(context_lines)
        
        return context, {
            "mode": "HybridRAG",
            "topk_indices": topk_indices.tolist(),
            "bm25_top1": bm25_scores.argmax().item(),
            "memory_top1": memory_scores.argmax().item(),
            "embed_top1": embed_scores.argmax().item(),
            "fused_top1": fused_scores.argmax().item(),
            "memory_confidence": memory_confidence,
            "dynamic_weights": {
                "bm25": round(bm25_weight, 3),
                "memory": round(memory_weight, 3),
                "embed": round(embed_weight, 3)
            }
        }
    
    def online_learn(self, question: str, expected_answer: str):
        """Learn from Q&A pair (Online Learning)."""
        dim = self.embedder.target_dim
        
        query_emb = self.embedder(question)
        query_vec = query_emb.reshape(-1, dim).mean(dim=0, keepdim=True)
        
        answer_emb = self.embedder(expected_answer)
        answer_vec = answer_emb.reshape(-1, dim).mean(dim=0, keepdim=True)
        
        self.titan_rag.titan.ltm.forward_with_update(query_vec, answer_vec)


class AdaptiveHybridRAG(BaseRetriever):
    """
    Hybrid RAG v2: Per-Chunk Soft Routing.
    
    For each chunk, computes BOTH a RAG score (BM25 + Embedding) and a
    Titan Memory score. Uses the RAG confidence scorer to produce a
    per-chunk blending weight:
    
        final_score[i] = conf[i] * rag_score[i] + (1 - conf[i]) * titan_score[i]
    
    Unlike V1 (HybridRAG) which uses a single global memory confidence to
    adjust weights for ALL chunks equally, V2 assigns each chunk its own
    RAG-vs-Titan ratio based on the trained scorer model.
    """
    
    def __init__(self, embedder, llm_generator, titan_rag, scorer=None,
                 base_bm25_weight: float = 0.4,
                 base_embed_weight: float = 0.6,
                 multihop: bool = False):
        super().__init__(embedder, llm_generator, multihop=multihop)
        self.titan_rag = titan_rag
        self.scorer = scorer
        self.bm25_weight = base_bm25_weight
        self.embed_weight = base_embed_weight
        
    def retrieve(self, question: str, documents: List[str],
                 doc_embeddings: torch.Tensor, topk: int = 5) -> Tuple[str, Dict]:
        """
        Per-chunk soft routing retrieval.
        
        1. Compute RAG scores (BM25 + Embedding) for ALL chunks
        2. Compute Titan scores (Memory) for ALL chunks
        3. Get per-chunk confidence from scorer
        4. Blend: final = conf * rag + (1 - conf) * titan
        5. Top-K
        """
        dim = self.embedder.target_dim
        
        # Embed question
        query_emb = self.embedder(question)
        query_flat = query_emb.reshape(-1, dim)
        query_vec = query_flat.mean(dim=0)
        
        # --- STEP 1: Per-chunk confidence from scorer ---
        if self.scorer is not None:
            questions_repeated = [question] * len(documents)
            confidence_np = self.scorer.predict_batch(questions_repeated, documents)
            confidence = torch.tensor(confidence_np, dtype=torch.float32,
                                      device=doc_embeddings.device)
        else:
            # Fallback: equal blend (0.5 RAG + 0.5 Titan)
            confidence = torch.full((len(documents),), 0.5,
                                    device=doc_embeddings.device)
        
        with torch.no_grad():
            # --- STEP 2: RAG scores (BM25 + Embedding) for ALL chunks ---
            bm25_scores = compute_bm25_scores(question, documents)
            bm25_scores = bm25_scores.to(doc_embeddings.device)
            embed_scores = F.cosine_similarity(query_vec.unsqueeze(0), doc_embeddings)
            
            rag_scores = (
                self.bm25_weight * normalize_scores(bm25_scores) +
                self.embed_weight * normalize_scores(embed_scores)
            )
            
            # --- STEP 3: Titan scores (Memory) for ALL chunks ---
            memory_output = self.titan_rag.titan.ltm.forward_no_update(query_flat)
            memory_vec = memory_output.mean(dim=0)
            titan_scores = F.cosine_similarity(memory_vec.unsqueeze(0), doc_embeddings)
            
            # --- STEP 4: Per-chunk soft fusion ---
            final_scores = (
                confidence * normalize_scores(rag_scores) +
                (1 - confidence) * normalize_scores(titan_scores)
            )
            
            # --- STEP 5: Top-K ---
            topk_values, topk_indices = final_scores.topk(min(topk, len(documents)))
        
        # Build context
        context_lines = [documents[idx] for idx in topk_indices.tolist()]
        context = "\n\n".join(context_lines)
        
        # Per-chunk confidence stats for selected chunks
        selected_conf = confidence[topk_indices].tolist()
        avg_conf = confidence.mean().item()
        
        return context, {
            "mode": "AdaptiveHybridRAG",
            "topk_indices": topk_indices.tolist(),
            "topk_confidence": [round(c, 3) for c in selected_conf],
            "avg_confidence": round(avg_conf, 3),
            "rag_top1": rag_scores.argmax().item(),
            "titan_top1": titan_scores.argmax().item(),
            "fused_top1": final_scores.argmax().item()
        }
    
    def online_learn(self, question: str, expected_answer: str):
        """Learn from Q&A pair (Online Learning).
        
        Updates Titan Memory with the question-answer association,
        improving future memory-guided retrieval for similar questions.
        """
        dim = self.embedder.target_dim
        
        query_emb = self.embedder(question)
        query_vec = query_emb.reshape(-1, dim).mean(dim=0, keepdim=True)
        
        answer_emb = self.embedder(expected_answer)
        answer_vec = answer_emb.reshape(-1, dim).mean(dim=0, keepdim=True)
        
        self.titan_rag.titan.ltm.forward_with_update(query_vec, answer_vec)


def create_retrievers(embedder, llm_generator, titan_rag=None, multihop=False,
                      scorer=None) -> Dict[str, BaseRetriever]:
    """
    Factory function to create all retriever instances.
    
    Args:
        embedder: SentenceTransformerEmbedder instance
        llm_generator: FlanT5Generator instance
        titan_rag: TitanRAG instance (required for TitanOnly and HybridRAG)
        multihop: If True, use multi-hop reasoning prompt (for HotpotQA)
        scorer: RAGConfidenceScorer instance (optional, for AdaptiveHybridRAG)
    
    Returns:
        dict: retriever name -> retriever instance
    """
    retrievers = {
        "pure_rag": PureRAG(embedder, llm_generator, multihop=multihop)
    }
    
    if titan_rag is not None:
        retrievers["titan_only"] = TitanOnly(embedder, llm_generator, titan_rag, multihop=multihop)
        retrievers["hybrid"] = HybridRAG(embedder, llm_generator, titan_rag, multihop=multihop)
        retrievers["adaptive"] = AdaptiveHybridRAG(embedder, llm_generator, titan_rag,
                                               scorer=scorer, multihop=multihop)
    
    return retrievers
