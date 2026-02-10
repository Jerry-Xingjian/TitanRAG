#!/usr/bin/env python3
"""
Baseline Comparison Script for TitanRAG.

Compares three retrieval strategies:
- PureRAG: BM25 + Embedding (no Memory)
- TitanOnly: Memory-guided retrieval only  
- HybridRAG: BM25 + Memory + Embedding fusion

Usage:
    # Use sample essays (original mode)
    python compare_baselines.py --essay climate --epochs 50
    
    # Use SQuAD dataset (single-doc per title)
    python compare_baselines.py --squad --titles 10 --epochs 50
    
    # Multi-document: shared memory across articles
    python compare_baselines.py --multi-doc --group-size 5 --epochs 50
"""

import argparse
import sys
import os
import re
import random
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))

from common.embedders import SentenceTransformerEmbedder
from common.titan_utils import create_titan_rag
from common.llm_utils import FlanT5Generator
from common.text_utils import split_into_chunks
from baselines import create_retrievers


# Import essays from data directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'))


def load_sample_essays():
    """Load sample essays for original mode."""
    from sample_essays import get_all_essays, get_test_questions
    return get_all_essays(), get_test_questions()


def load_squad_data():
    """Load processed SQuAD data."""
    try:
        from processed_squad import get_all_contexts, get_test_questions
        return get_all_contexts(), get_test_questions()
    except ImportError:
        print("❌ Error: processed_squad.py not found. Please run script.py first.")
        sys.exit(1)


def extract_key_elements(text):
    """Extract key numbers and keywords from text for matching."""
    text = text.lower()
    numbers = re.findall(r'[\d.]+%?', text)
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
    
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return set(keywords)


def evaluate_answer_quality(expected, got):
    """Flexible answer evaluation using multiple criteria."""
    expected_lower = expected.lower()
    got_lower = got.lower()
    
    # 1. Bidirectional containment
    if expected_lower in got_lower or got_lower in expected_lower:
        return True
    
    # 2. Phrase overlap
    got_words = got_lower.split()
    if len(got_words) >= 3:
        for i in range(len(got_words) - 2):
            phrase = ' '.join(got_words[i:i+3])
            if phrase in expected_lower:
                return True
    
    # 3. Key elements matching
    expected_keys = extract_key_elements(expected)
    got_keys = extract_key_elements(got)
    if expected_keys and expected_keys & got_keys:
        return True
    
    # 4. Keyword overlap >= 40%
    expected_keywords = extract_keywords(expected)
    got_keywords = extract_keywords(got)
    if expected_keywords:
        overlap = len(expected_keywords & got_keywords)
        if overlap / len(expected_keywords) >= 0.4:
            return True
    
    return False


def evaluate_retriever(retriever, documents, doc_embeddings, questions, verbose=True, show_progress=False):
    """Evaluate a retriever on a set of questions."""
    correct = 0
    total = len(questions)
    
    results = []
    progress_chars = []
    
    for i, (question, expected) in enumerate(questions):
        result = retriever.answer(question, documents, doc_embeddings, topk=3)
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
        
        # Show progress indicator
        if show_progress:
            status_char = "." if is_correct else "x"
            progress_chars.append(status_char)
            # Print progress every 5 questions or at the end
            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"      [{i+1}/{total}] {''.join(progress_chars[-10:])}", end="\r")
        
        if verbose:
            status = "ok" if is_correct else "X"
            print(f"      [{i+1}/{total}] {status} Q: {question[:50]}")
            print(f"           Exp: {expected[:40]}")
            print(f"           Got: {result['answer'][:40]}")
    
    if show_progress:
        print()  # New line after progress
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, results


def run_sample_essay_mode(args):
    """Run comparison on sample essays (original mode)."""
    ESSAYS, TEST_QUESTIONS = load_sample_essays()
    
    print("=" * 60)
    print("BASELINE COMPARISON: PureRAG vs TitanOnly vs HybridRAG")
    print(f"Mode: Sample Essay ({args.essay})")
    print("=" * 60)
    
    # Initialize components
    print("\n📦 Loading components...")
    
    # Import DEVICE from embedders or titan_utils
    from common.embedders import DEVICE
    
    embedder = SentenceTransformerEmbedder(target_dim=256, device=DEVICE)
    llm = FlanT5Generator("google/flan-t5-large", device=DEVICE)
    titan_rag = create_titan_rag(dim=256, device=DEVICE)
    
    # Load essay
    text = ESSAYS[args.essay]
    questions = TEST_QUESTIONS[args.essay]
    
    # Parse into semantic chunks
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
    
    # Create retrievers and evaluate
    retrievers = create_retrievers(embedder, llm, titan_rag)
    
    print("\n" + "=" * 60)
    results_summary = {}
    
    for name, retriever in retrievers.items():
        print(f"\n📊 Evaluating: {name.upper()}")
        print("-" * 40)
        accuracy, results = evaluate_retriever(retriever, chunks, chunk_embeddings, questions)
        results_summary[name] = accuracy
        print(f"\n   Accuracy: {accuracy:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, accuracy in results_summary.items():
        bar = "█" * int(accuracy / 10) + "░" * (10 - int(accuracy / 10))
        print(f"  {name:12s}: {bar} {accuracy:.1f}%")


def run_squad_mode(args):
    """Run comparison on SQuAD dataset (new mode)."""
    CONTEXTS, TEST_QUESTIONS = load_squad_data()
    
    print("=" * 60)
    print("BASELINE COMPARISON: PureRAG vs TitanOnly vs HybridRAG")
    print(f"Mode: SQuAD Dataset ({args.titles} titles)")
    print("=" * 60)
    
    # Initialize components
    print("\n📦 Loading components...")
    
    from common.embedders import DEVICE
    
    embedder = SentenceTransformerEmbedder(target_dim=256, device=DEVICE)
    llm = FlanT5Generator("google/flan-t5-large", device=DEVICE)
    
    # Select titles to evaluate
    all_titles = list(CONTEXTS.keys())
    if args.titles >= len(all_titles):
        selected_titles = all_titles
    else:
        # Random sample
        random.seed(42)  # For reproducibility
        selected_titles = random.sample(all_titles, args.titles)
    
    print(f"   Total titles available: {len(all_titles)}")
    print(f"   Selected for evaluation: {len(selected_titles)}")
    
    # Aggregate results
    all_results = {
        "pure_rag": {"correct": 0, "total": 0},
        "titan_only": {"correct": 0, "total": 0},
        "hybrid": {"correct": 0, "total": 0}
    }
    
    # Process each title
    for i, title in enumerate(selected_titles):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(selected_titles)}] Title: {title[:50]}...")
        print("=" * 60)
        
        context = CONTEXTS[title]
        questions = TEST_QUESTIONS.get(title, [])
        
        if not questions:
            print("   ⚠️ No questions for this title, skipping...")
            continue
        
        # Limit questions per title to avoid long runs
        if len(questions) > args.max_questions:
            questions = questions[:args.max_questions]
        
        print(f"   Context length: {len(context)} chars")
        print(f"   Questions: {len(questions)}")
        
        # Create fresh Titan model for each title
        titan_rag = create_titan_rag(dim=256)
        
        # Parse context into chunks (fixed size for stability)
        chunks = split_into_chunks(context, sentences_per_chunk=3, overlap_sentences=1)
        if not chunks:
            # Fallback: split by paragraphs
            chunks = [p.strip() for p in context.split('\n\n') if p.strip()]
        if not chunks:
            chunks = [context]
        
        print(f"   Chunks: {len(chunks)}")
        
        # Embed chunks
        chunk_embeddings = embedder.embed_batch(chunks)
        
        # Digest document (fixed epochs for stability)
        digest_epochs = min(args.epochs, 500)
        print(f"   Digesting ({digest_epochs} epochs): ", end="", flush=True)
        for epoch in range(digest_epochs):
            all_emb = embedder.embed_batch(chunks)
            titan_rag.digest_knowledge(all_emb.unsqueeze(0))
            if (epoch + 1) % 10 == 0:
                print(f"{epoch+1}", end=" ", flush=True)
        print("done")
        
        # Create retrievers
        retrievers = create_retrievers(embedder, llm, titan_rag)
        
        # Evaluate each retriever with progress
        for name, retriever in retrievers.items():
            print(f"   [{name}] Evaluating...", end="", flush=True)
            accuracy, results = evaluate_retriever(
                retriever, chunks, chunk_embeddings, questions, 
                verbose=args.verbose, show_progress=not args.verbose
            )
            
            correct_count = sum(1 for r in results if r["correct"])
            all_results[name]["correct"] += correct_count
            all_results[name]["total"] += len(questions)
            
            print(f" {correct_count}/{len(questions)} ({accuracy:.1f}%)")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Evaluated on {len(selected_titles)} titles")
    print()
    
    for name, stats in all_results.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"] * 100
            bar = "█" * int(accuracy / 10) + "░" * (10 - int(accuracy / 10))
            print(f"  {name:12s}: {bar} {accuracy:.1f}% ({stats['correct']}/{stats['total']})")


def run_multidoc_mode(args):
    """Run comparison with multiple articles digested into shared memory."""
    CONTEXTS, TEST_QUESTIONS = load_squad_data()

    print("=" * 60)
    print("BASELINE COMPARISON: Multi-Document Cross-Article Retrieval")
    print(f"Mode: SQuAD Multi-Doc (group_size={args.group_size})")
    print("=" * 60)

    # Initialize components
    print("\n📦 Loading components...")
    from common.embedders import DEVICE

    embedder = SentenceTransformerEmbedder(target_dim=256, device=DEVICE)
    llm = FlanT5Generator("google/flan-t5-large", device=DEVICE)

    # Select titles
    all_titles = list(CONTEXTS.keys())
    random.seed(42)
    if args.group_size >= len(all_titles):
        selected_titles = all_titles
    else:
        selected_titles = random.sample(all_titles, args.group_size)

    print(f"   Total titles available: {len(all_titles)}")
    print(f"   Selected for multi-doc group: {len(selected_titles)}")
    for t in selected_titles:
        print(f"     - {t}")

    # ---- Build global chunk pool ----
    print("\n📄 Building global chunk pool...")
    all_chunks = []          # global list of text chunks
    chunk_title_map = []     # parallel list: which title each chunk belongs to
    all_questions = []       # (title, question, expected_answer)

    for title in selected_titles:
        context = CONTEXTS[title]
        chunks = split_into_chunks(context, sentences_per_chunk=3, overlap_sentences=1)
        if not chunks:
            chunks = [p.strip() for p in context.split('\n\n') if p.strip()]
        if not chunks:
            chunks = [context]

        all_chunks.extend(chunks)
        chunk_title_map.extend([title] * len(chunks))

        questions = TEST_QUESTIONS.get(title, [])
        if questions:
            if len(questions) > args.max_questions:
                questions = questions[:args.max_questions]
            for q, a in questions:
                all_questions.append((title, q, a))

    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Total questions: {len(all_questions)}")

    # ---- Embed all chunks ----
    print("\n🔢 Embedding all chunks...")
    all_embeddings = embedder.embed_batch(all_chunks)

    # ---- Shared Titan memory: digest ALL chunks ----
    titan_rag = create_titan_rag(dim=256, device=DEVICE)
    digest_epochs = min(args.epochs, 500)
    print(f"\n🧠 Digesting {len(all_chunks)} chunks into shared memory ({digest_epochs} epochs)...")
    for epoch in range(digest_epochs):
        emb = embedder.embed_batch(all_chunks)
        titan_rag.digest_knowledge(emb.unsqueeze(0))
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{digest_epochs}")

    # ---- Create retrievers (shared memory) ----
    retrievers = create_retrievers(embedder, llm, titan_rag)

    # ---- Evaluate ----
    print("\n" + "=" * 60)
    print("EVALUATING (cross-document retrieval)")
    print("=" * 60)

    # Prepare questions in evaluate_retriever format: list of (question, expected)
    eval_questions = [(q, a) for (_, q, a) in all_questions]

    # Per-title tracking
    title_results = {name: {} for name in retrievers}  # {retriever: {title: [correct_bools]}}
    for name in retrievers:
        for title in selected_titles:
            title_results[name][title] = []

    results_summary = {}
    for name, retriever in retrievers.items():
        print(f"\n📊 Evaluating: {name.upper()}")
        print("-" * 40)
        accuracy, results = evaluate_retriever(
            retriever, all_chunks, all_embeddings, eval_questions,
            verbose=args.verbose, show_progress=not args.verbose
        )
        results_summary[name] = accuracy

        # Map results back to titles for per-title stats
        for idx, (title, q, a) in enumerate(all_questions):
            title_results[name][title].append(results[idx]["correct"])

        print(f"   Overall Accuracy: {accuracy:.1f}%")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    for name, accuracy in results_summary.items():
        bar = "█" * int(accuracy / 10) + "░" * (10 - int(accuracy / 10))
        print(f"  {name:12s}: {bar} {accuracy:.1f}%")

    # Per-title breakdown
    print("\n" + "-" * 60)
    print("PER-TITLE BREAKDOWN")
    print("-" * 60)
    header = f"  {'Title':<30s}"
    for name in retrievers:
        header += f" {name:>12s}"
    print(header)
    print("  " + "-" * (30 + 13 * len(retrievers)))

    for title in selected_titles:
        row = f"  {title[:30]:<30s}"
        for name in retrievers:
            bools = title_results[name][title]
            if bools:
                acc = sum(bools) / len(bools) * 100
                row += f" {acc:>10.1f}% "
            else:
                row += f" {'N/A':>11s} "
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Compare baseline retrieval strategies")

    # Mode selection
    parser.add_argument("--squad", action="store_true",
                       help="Use SQuAD dataset instead of sample essays")
    parser.add_argument("--multi-doc", action="store_true",
                       help="Multi-document mode: digest multiple articles into shared memory")

    # Sample essay mode options
    parser.add_argument("--essay", type=str, default="climate",
                       help="Essay to use (climate/ai/space)")

    # SQuAD mode options
    parser.add_argument("--titles", type=int, default=10,
                       help="Number of titles to evaluate (SQuAD mode)")
    parser.add_argument("--max-questions", type=int, default=5,
                       help="Max questions per title (SQuAD/multi-doc mode)")

    # Multi-doc mode options
    parser.add_argument("--group-size", type=int, default=5,
                       help="Number of titles to group together (multi-doc mode)")

    # Common options
    parser.add_argument("--epochs", type=int, default=50, help="Digestion epochs")
    parser.add_argument("--topk", type=int, default=5, help="Top-K for retrieval")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    if args.multi_doc:
        run_multidoc_mode(args)
    elif args.squad:
        run_squad_mode(args)
    else:
        run_sample_essay_mode(args)


if __name__ == "__main__":
    main()
