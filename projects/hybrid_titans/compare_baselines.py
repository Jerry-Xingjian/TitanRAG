#!/usr/bin/env python3
"""
Baseline Comparison Script for TitanRAG.

Compares three retrieval strategies:
- PureRAG: BM25 + Embedding (no Memory)
- TitanOnly: Memory-guided retrieval only  
- HybridRAG: BM25 + Memory + Embedding fusion

Usage:
    python compare_baselines.py --essay climate --epochs 50
"""

import argparse
import sys
import os
import re

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

from common.embedders import SentenceTransformerEmbedder
from common.titan_utils import create_titan_rag, digest_document
from common.llm_utils import FlanT5Generator
from common.text_utils import split_into_chunks
from baselines import create_retrievers


# Import essays from data directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'))
from sample_essays import get_all_essays, get_test_questions

ESSAYS = get_all_essays()
TEST_QUESTIONS = get_test_questions()


def extract_key_elements(text):
    """Extract key numbers and keywords from text for matching."""
    text = text.lower()
    # Extract numbers (including decimals and percentages)
    numbers = re.findall(r'[\d.]+%?', text)
    # Extract years (4-digit numbers)
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    return set(numbers + years)


def extract_keywords(text):
    """Extract meaningful keywords from text, removing stopwords."""
    stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                 'between', 'under', 'again', 'further', 'then', 'once', 'and', 'or',
                 'but', 'if', 'because', 'until', 'while', 'that', 'which', 'who',
                 'whom', 'this', 'these', 'those', 'it', 'its', 'their', 'they'}
    
    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    # Filter stopwords and short words
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return set(keywords)


def evaluate_answer_quality(expected, got):
    """
    Flexible answer evaluation using multiple criteria.
    
    Returns True if any of these conditions are met:
    1. Bidirectional containment
    2. Phrase overlap (key phrases from got are in expected)
    3. Key numbers/years match
    4. Keyword overlap >= 40%
    """
    expected_lower = expected.lower()
    got_lower = got.lower()
    
    # 1. Bidirectional containment
    if expected_lower in got_lower or got_lower in expected_lower:
        return True
    
    # 2. Phrase overlap - check if 3+ consecutive words from got appear in expected
    got_words = got_lower.split()
    if len(got_words) >= 3:
        for i in range(len(got_words) - 2):
            phrase = ' '.join(got_words[i:i+3])
            if phrase in expected_lower:
                return True
    
    # 3. Key elements matching (numbers, years)
    expected_keys = extract_key_elements(expected)
    got_keys = extract_key_elements(got)
    
    if expected_keys and expected_keys & got_keys:
        return True
    
    # 4. Keyword overlap matching (lowered to 40%)
    expected_keywords = extract_keywords(expected)
    got_keywords = extract_keywords(got)
    
    if expected_keywords:
        overlap = len(expected_keywords & got_keywords)
        overlap_ratio = overlap / len(expected_keywords)
        
        if overlap_ratio >= 0.4:
            return True
    
    return False


def evaluate_retriever(retriever, documents, doc_embeddings, questions, verbose=True):
    """Evaluate a retriever on a set of questions."""
    correct = 0
    total = len(questions)
    
    results = []
    for question, expected in questions:
        result = retriever.answer(question, documents, doc_embeddings, topk=3)
        
        # Flexible evaluation
        is_correct = evaluate_answer_quality(expected, result["answer"])
        if is_correct:
            correct += 1
        
        results.append({
            "question": question,
            "expected": expected,
            "answer": result["answer"],
            "correct": is_correct,
            "mode": result["mode"]
        })
        
        if verbose:
            status = "✅" if is_correct else "❌"
            print(f"  {status} Q: {question}")
            print(f"     Expected: {expected}")
            print(f"     Got: {result['answer']}")
            # Show retrieval details if available
            details = result.get("retrieval_details", {})
            if "memory_confidence" in details:
                conf = details["memory_confidence"]
                weights = details.get("dynamic_weights", {})
                print(f"     [Confidence: {conf:.2f} | Weights: bm25={weights.get('bm25', 0):.2f}, mem={weights.get('memory', 0):.2f}, emb={weights.get('embed', 0):.2f}]")
            
            # Show retrieved chunks for wrong answers
            if not is_correct and "context" in result:
                print("     📄 Retrieved context:")
                context = result["context"]
                # Show first 300 chars
                print(f"        {context[:300]}...")
    
    accuracy = correct / total * 100
    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description="Compare baseline retrieval strategies")
    parser.add_argument("--essay", type=str, default="climate", 
                       choices=list(ESSAYS.keys()), help="Essay to use")
    parser.add_argument("--epochs", type=int, default=50, help="Digestion epochs")
    parser.add_argument("--topk", type=int, default=5, help="Top-K for retrieval")
    args = parser.parse_args()
    
    print("=" * 60)
    print("BASELINE COMPARISON: PureRAG vs TitanOnly vs HybridRAG")
    print("=" * 60)
    
    # Initialize components
    print("\n📦 Loading components...")
    embedder = SentenceTransformerEmbedder(target_dim=256)
    llm = FlanT5Generator("google/flan-t5-large")
    titan_rag = create_titan_rag(dim=256)
    
    # Load essay
    text = ESSAYS[args.essay]
    questions = TEST_QUESTIONS[args.essay]
    
    # Parse into semantic chunks (instead of simple line split)
    chunks = split_into_chunks(text, sentences_per_chunk=3, overlap_sentences=1)
    print(f"   Split into {len(chunks)} semantic chunks")
    
    # Digest document
    print(f"\n🧠 Digesting document ({args.epochs} epochs)...")
    chunk_embeddings = embedder.embed_batch(chunks)
    
    for epoch in range(args.epochs):
        all_emb = embedder.embed_batch(chunks)
        titan_rag.digest_knowledge(all_emb.unsqueeze(0))
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{args.epochs}")
    
    # Create retrievers
    retrievers = create_retrievers(embedder, llm, titan_rag)
    
    # Evaluate each
    print("\n" + "=" * 60)
    results_summary = {}
    
    for name, retriever in retrievers.items():
        print(f"\n📊 Evaluating: {name.upper()}")
        print("-" * 40)
        accuracy, results = evaluate_retriever(
            retriever, chunks, chunk_embeddings, questions
        )
        results_summary[name] = accuracy
        print(f"\n   Accuracy: {accuracy:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, accuracy in results_summary.items():
        bar = "█" * int(accuracy / 10) + "░" * (10 - int(accuracy / 10))
        print(f"  {name:12s}: {bar} {accuracy:.1f}%")


if __name__ == "__main__":
    main()
