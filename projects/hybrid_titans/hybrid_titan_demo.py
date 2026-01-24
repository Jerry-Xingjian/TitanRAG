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

# Add parent directory to path (hybrid_titans -> projects -> TitanLLM)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

from main import TitanRAG
from data.sample_essays import ESSAY_CLIMATE, ESSAY_AI, ESSAY_SPACE, TEST_QUESTIONS


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


class HybridTitanRAG(nn.Module):
    """
    Hybrid Titan: TitanRAG Retrieval + LLM Generation
    
    This combines:
    1. TitanRAG for semantic memory-based retrieval
    2. External LLM (Flan-T5-2) for answer generation
    """
    
    def __init__(self, titan_rag, embedder, llm_name="google/flan-t5-large"):
        super().__init__()
        self.titan_rag = titan_rag
        self.embedder = embedder
        
        # Load Flan-T5 for generation (better instruction following than Flan-T5-2)
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(llm_name)
        self.llm = T5ForConditionalGeneration.from_pretrained(llm_name)
        
        print(f"✅ Loaded LLM: {llm_name}")
    
    def retrieve(self, question, essay_lines, line_embeddings, topk=5):
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
        # Build prompt optimized for Flan-T5 instruction format
        prompt = f"""Based on the following context, answer the question with specific facts and numbers.

Context: {context}

Question: {question}

Answer:"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Generate (greedy decoding for more accurate extraction)
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,  # Beam search for better quality
                early_stopping=True
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer, prompt
    
    def answer(self, question, essay_lines, line_embeddings, expected_answer=None, topk=5, max_new_tokens=100):
        """
        Full pipeline: Retrieve context, generate answer, and learn from Q&A.
        
        Args:
            question: The question to answer
            essay_lines: List of essay lines
            line_embeddings: Tensor of line embeddings
            expected_answer: If provided, model learns from this Q&A pair (Online Learning)
            topk: Number of context lines to retrieve
            max_new_tokens: Max tokens to generate
        """
        dim = self.embedder.target_dim
        
        # Embed question
        query_emb = self.embedder(question)
        query_flat = query_emb.reshape(-1, dim)
        
        # Step 1: Retrieve
        context, retrieval_details = self.retrieve(
            question, essay_lines, line_embeddings, topk
        )
        
        # Step 2: Generate
        answer, prompt = self.generate(question, context, max_new_tokens)
        
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
            "prompt": prompt,
            "online_learned": learned
        }


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
    
    # Parse essay into PARAGRAPHS (not lines) for better context retention
    # Split by double newlines, then clean up each paragraph
    raw_paragraphs = essay_text.split('\n\n')
    essay_paragraphs = []
    for para in raw_paragraphs:
        # Merge lines within paragraph, clean up
        cleaned = ' '.join(line.strip() for line in para.split('\n') if line.strip())
        if cleaned and len(cleaned) > 10:  # Skip empty or very short
            essay_paragraphs.append(cleaned)
    
    print(f"\n🔄 Embedding {len(essay_paragraphs)} paragraphs...")
    
    # Embed all paragraphs
    with torch.no_grad():
        para_embeddings = []
        for para in essay_paragraphs:
            para_emb = embedder(para)
            para_vec = para_emb.reshape(-1, args.dim).mean(dim=0)
            para_embeddings.append(para_vec)
        line_embeddings = torch.stack(para_embeddings)  # Keep variable name for compatibility
    
    # Use paragraphs instead of lines
    essay_lines = essay_paragraphs  # Alias for compatibility
    
    # Digest into TitanRAG
    print(f"\n🧠 Digesting into TitanRAG ({args.epochs} epochs)...")
    all_embeddings = torch.cat([embedder(line) for line in essay_lines], dim=1)
    
    for epoch in range(args.epochs):
        titan_rag.digest_knowledge(all_embeddings)
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{args.epochs} completed")
    
    # Initialize Hybrid Titan with Flan-T5
    print("\n🤖 Initializing Hybrid Titan (Flan-T5 generator)...")
    hybrid_titan = HybridTitanRAG(titan_rag, embedder)  # Uses default Flan-T5
    
    # Run Q&A with Online Learning
    print("\n" + "=" * 60)
    print("QUESTION ANSWERING with ONLINE LEARNING")
    print("=" * 60)
    
    for i, (question, expected) in enumerate(essay_questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Q{i}: {question}")
        print(f"Expected: {expected}")
        
        # Get answer from Hybrid Titan (with Online Learning!)
        result = hybrid_titan.answer(
            question, 
            essay_lines, 
            line_embeddings,
            expected_answer=expected,  # Enable Online Learning
            topk=args.topk,
            max_new_tokens=args.max_tokens
        )
        
        print(f"\n[Retrieval] Indices: {result['retrieval']['topk_indices']}")
        print(f"[Context]\n  {result['context']}")
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
    
    print("\n" + "=" * 60)
    print("✅ Hybrid Titan Demo Complete!")
    print("=" * 60)
    
    return hybrid_titan


def main():
    parser = argparse.ArgumentParser(description="Hybrid Titan Demo")
    parser.add_argument("--essay", type=str, default="climate",
                       choices=["climate", "ai", "space"],
                       help="Essay to use")
    parser.add_argument("--dim", type=int, default=256,
                       help="Embedding dimension")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Digestion epochs")
    parser.add_argument("--topk", type=int, default=3,
                       help="Top-k lines to retrieve")
    parser.add_argument("--max_tokens", type=int, default=50,
                       help="Max tokens to generate")
    
    args = parser.parse_args()
    run_hybrid_titan_demo(args)


if __name__ == "__main__":
    main()
