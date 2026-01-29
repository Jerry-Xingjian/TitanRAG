"""
Pure Titan Demo:

This experiment uses ONLY Titans memory for question answering.
- Document is compressed into Titan long-term memory (LTM)
- Questions query the memory array directly
- Flan-T5 generates answers conditioned on retrieved memory vectors

Architecture:
    Document → Titan Memory Writes → Memory Array
    Question → Titan Memory Read → Soft Prompt → FLAN-T5 → Answer
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


class PureTitanRAG(nn.Module):
    """
    Pure Titan QA: Titan Memory + LLM Generation

    No RAG. No paragraph retrieval.
    All knowledge comes from Titan long-term memory (LTM).
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

        # Project Titan memory vectors into T5 embedding space (soft prompt)
        self.memory_to_t5 = nn.Linear(embedder.target_dim, self.llm.config.d_model)
    
    def generate(self, question, memory_vec, max_new_tokens=100):
        """
        Generate answer using Flan-T5 conditioned on Titan memory (no text context).
        """
        # Convert memory vector into soft prompt token
        prefix = self.memory_to_t5(memory_vec)          # (1, d_model)
        prefix = prefix.unsqueeze(1)                    # (1, 1, d_model)

        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        input_embeds = self.llm.encoder.embed_tokens(inputs.input_ids)

        # Prepend memory prefix
        inputs_embeds = torch.cat([prefix, input_embeds], dim=1)

        attention_mask = torch.cat([
            torch.ones((1, 1), dtype=torch.long),
            inputs.attention_mask
        ], dim=1)

        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                early_stopping=True
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def answer(self, question, expected_answer=None, max_new_tokens=100):
        """
        Pure Titans QA:
        1. Question queries Titan memory
        2. Memory vector conditions generation
        3. Optional online learning stores Q→A into memory
        """
        dim = self.embedder.target_dim

        # Embed question
        query_emb = self.embedder(question)
        query_vec = query_emb.reshape(-1, dim)

        # Step 1: Query Titan memory (no update)
        with torch.no_grad():
            memory_output = self.titan_rag.titan.ltm.forward_no_update(query_vec)
            memory_vec = memory_output.mean(dim=0, keepdim=True)

        # Step 2: Generate using memory only
        answer = self.generate(question, memory_vec, max_new_tokens)

        # Step 3: Online learning (store Q → A in memory)
        learned = False
        if expected_answer is not None:
            expected_emb = self.embedder(expected_answer)
            expected_vec = expected_emb.reshape(-1, dim).mean(dim=0, keepdim=True)

            self.titan_rag.titan.ltm.forward_with_update(
                query_vec.mean(dim=0, keepdim=True),
                expected_vec
            )
            learned = True

        return {
            "answer": answer,
            "memory_used": True,
            "online_learned": learned
        }


def run_pure_titan_demo(args):
    """Run the Pure Titan demo."""
    
    print("=" * 60)
    print("PURE TITAN MEMORY DEMO (NO RAG)")
    print("Titan Memory → Flan-T5 Generation")
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
    
    # Digest into TitanRAG
    print(f"\n🧠 Digesting into TitanRAG ({args.epochs} epochs)...")
    all_embeddings = torch.cat([embedder(p) for p in essay_paragraphs], dim=1)
    
    for epoch in range(args.epochs):
        titan_rag.digest_knowledge(all_embeddings)
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{args.epochs} completed")
    
    # Initialize Pure Titan with Flan-T5
    print("\n🤖 Initializing Pure Titan (Flan-T5 generator)...")
    pure_titan = PureTitanRAG(titan_rag, embedder)  # Uses default Flan-T5
    
    # Run Q&A with Online Learning
    print("\n" + "=" * 60)
    print("QUESTION ANSWERING with ONLINE LEARNING")
    print("=" * 60)
    
    for i, (question, expected) in enumerate(essay_questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Q{i}: {question}")
        print(f"Expected: {expected}")
        
        # Get answer from Pure Titan (with Online Learning!)
        result = pure_titan.answer(
            question,
            expected_answer=expected,
            max_new_tokens=args.max_tokens
        )
        
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
    print("✅ Pure Titan Memory Demo Complete!")
    print("=" * 60)
    
    return pure_titan


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
    run_pure_titan_demo(args)


if __name__ == "__main__":
    main()
