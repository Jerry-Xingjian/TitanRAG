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
                 fallback_confidence_threshold=0.3, enable_fallback=True):
        super().__init__()
        self.titan_rag = titan_rag
        self.embedder = embedder
        
        # Load Flan-T5 for generation (better instruction following than Flan-T5-2)
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(llm_name)
        self.llm = T5ForConditionalGeneration.from_pretrained(llm_name)
        
        # Initialize Decision Maker
        self.decider = HybridDecider(
            question_weight=question_weight,
            memory_weight=memory_weight,
            dispersion_weight=dispersion_weight,
            hybrid_threshold=decision_threshold
        )
        
        # Fallback settings: if titans_only confidence is low, fallback to hybrid
        self.fallback_confidence_threshold = fallback_confidence_threshold
        self.enable_fallback = enable_fallback
        
        print(f"✅ Loaded LLM: {llm_name}")
        print(f"✅ Initialized HybridDecider (threshold={decision_threshold})")
        if enable_fallback:
            print(f"✅ Fallback enabled (confidence < {fallback_confidence_threshold} → switch to hybrid)")
    
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
    
    def generate_from_memory(self, question, query_emb, essay_lines, line_embeddings, max_new_tokens=100, topk=2):
        """
        Generate answer using Titan memory-guided retrieval (titans_only mode).
        
        Key improvement: Uses Titan LTM output to find relevant context,
        rather than letting the LLM guess blindly.
        
        Strategy:
        1. Get memory-enhanced representation from Titan LTM
        2. Use memory output to find most similar paragraphs (memory-guided retrieval)
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
            
            # Get top-k paragraphs based on memory similarity (not full ensemble fusion)
            topk_values, topk_indices = memory_scores.topk(min(topk, len(essay_lines)))
            
            # Build context from memory-recalled paragraphs
            context_lines = [essay_lines[idx] for idx in topk_indices.tolist()]
            memory_context = " ".join(context_lines)
        
        # Step 3: Build prompt with memory-recalled context
        # Use a simpler prompt since this is "memory-based" recall
        prompt = f"""Based on the following information from memory, answer the question.

Context from memory: {memory_context}

Question: {question}

Answer:"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                early_stopping=True
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
            "fallback_triggered": fallback_triggered
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
    
    # Initialize Hybrid Titan with Flan-T5 and Decision Maker
    print("\n🤖 Initializing Hybrid Titan (Flan-T5 generator + Decision Maker)...")
    hybrid_titan = HybridTitanRAG(
        titan_rag, 
        embedder,
        decision_threshold=args.decision_threshold,
        question_weight=args.question_weight,
        memory_weight=args.memory_weight,
        dispersion_weight=args.dispersion_weight,
        fallback_confidence_threshold=args.fallback_threshold,
        enable_fallback=args.enable_fallback
    )
    
    # Run Q&A with Decision Maker + Online Learning
    print("\n" + "=" * 60)
    print("QUESTION ANSWERING with DECISION MAKER + ONLINE LEARNING")
    print("=" * 60)
    print(f"Decision Threshold: {args.decision_threshold}")
    print(f"Weights - Question: {args.question_weight}, Memory: {args.memory_weight}, Dispersion: {args.dispersion_weight}")
    if args.enable_fallback:
        print(f"Fallback Enabled: titans_only → hybrid when confidence < {args.fallback_threshold}")
    
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
    parser.add_argument("--enable_fallback", type=bool, default=True,
                       help="Enable fallback from titans_only to hybrid when low confidence")
    parser.add_argument("--fallback_threshold", type=float, default=0.3,
                       help="Fallback confidence threshold (lower = more fallbacks)")
    
    args = parser.parse_args()
    run_hybrid_titan_demo(args)


if __name__ == "__main__":
    main()
