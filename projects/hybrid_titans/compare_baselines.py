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

    # Use HotpotQA dataset (multi-hop QA, already multi-document)
    python compare_baselines.py --hotpotqa --titles 10 --epochs 50
"""

import argparse
import sys
import os
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
from common.titan_utils import create_titan_rag, digest_chunks as _digest_chunks
from common.llm_utils import FlanT5Generator
from common.text_utils import chunk_context as _chunk_context
from common.eval_utils import evaluate_retriever
from common.output_utils import print_summary_bar as _print_summary_bar, save_results as _save_results
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
        print("❌ Error: processed_squad.py not found. Please run process_squad_data.py first.")
        sys.exit(1)


def load_hotpotqa_data():
    """Load processed HotpotQA data."""
    try:
        from processed_hotpotqa import get_all_contexts, get_test_questions
        return get_all_contexts(), get_test_questions()
    except ImportError:
        print("❌ Error: processed_hotpotqa.py not found. Please run process_hotpotqa_data.py first.")
        sys.exit(1)


def run_sample_essay_mode(args):
    """Run comparison on sample essays (original mode)."""
    ESSAYS, TEST_QUESTIONS = load_sample_essays()
    
    print("=" * 60)
    print("BASELINE COMPARISON: PureRAG vs TitanOnly vs HybridRAG")
    print(f"Mode: Sample Essay ({args.essay})")
    print("=" * 60)

    embedder, llm, DEVICE = _init_components()
    titan_rag = create_titan_rag(dim=256, device=DEVICE)

    # Chunk and digest essay
    text = ESSAYS[args.essay]
    questions = TEST_QUESTIONS[args.essay]
    chunks = _chunk_context(text)
    print(f"   Split into {len(chunks)} semantic chunks")

    chunk_embeddings = embedder.embed_batch(chunks)
    _digest_chunks(embedder, titan_rag, chunks, args.epochs, inline=False)

    # Create retrievers and evaluate
    retrievers = create_retrievers(embedder, llm, titan_rag)

    print("\n" + "=" * 60)
    results_summary = {}
    all_details = {}
    for name, retriever in retrievers.items():
        print(f"\n📊 Evaluating: {name.upper()}")
        print("-" * 40)
        accuracy, results = evaluate_retriever(retriever, chunks, chunk_embeddings, questions,
                                               verbose=args.verbose, show_progress=not args.verbose,
                                               embedder=embedder, topk=args.topk)
        results_summary[name] = accuracy
        all_details[name] = results
        print(f"\n   Accuracy: {accuracy:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    _print_summary_bar(results_summary, show_counts=False)

    if getattr(args, 'save_results', False):
        _save_results(f"essay_{args.essay}",
                      {"essay": args.essay, "epochs": args.epochs, "topk": args.topk},
                      results_summary,
                      details=all_details)


def _init_components():
    """Initialize shared components (embedder, LLM). Returns (embedder, llm, DEVICE)."""
    print("\n📦 Loading components...")
    from common.embedders import DEVICE
    embedder = SentenceTransformerEmbedder(target_dim=256, device=DEVICE)
    llm = FlanT5Generator("google/flan-t5-xl", device=DEVICE)
    return embedder, llm, DEVICE



def run_dataset_mode(args, contexts, questions, dataset_name):
    """
    Shared evaluation logic for dataset-based modes (SQuAD, HotpotQA, etc.).
    
    Each title gets its own Titan memory, chunks are digested independently,
    and all three retrievers are evaluated per-title.
    """
    print("=" * 60)
    print("BASELINE COMPARISON: PureRAG vs TitanOnly vs HybridRAG")
    print(f"Mode: {dataset_name} ({args.titles} titles)")
    print("=" * 60)

    embedder, llm, DEVICE = _init_components()

    # Select titles to evaluate
    all_titles = list(contexts.keys())
    if args.titles >= len(all_titles):
        selected_titles = all_titles
    else:
        random.seed(42)
        selected_titles = random.sample(all_titles, args.titles)

    print(f"   Total titles available: {len(all_titles)}")
    print(f"   Selected for evaluation: {len(selected_titles)}")

    all_results = {
        "pure_rag": {"correct": 0, "total": 0},
        "titan_only": {"correct": 0, "total": 0},
        "hybrid": {"correct": 0, "total": 0}
    }
    all_details = {name: {} for name in all_results}  # {retriever: {title: [results]}}

    for i, title in enumerate(selected_titles):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(selected_titles)}] Title: {title[:50]}...")
        print("=" * 60)

        context = contexts[title]
        title_questions = questions.get(title, [])

        if not title_questions:
            print("   ⚠️ No questions for this title, skipping...")
            continue

        if len(title_questions) > args.max_questions:
            title_questions = title_questions[:args.max_questions]

        print(f"   Context length: {len(context)} chars")
        print(f"   Questions: {len(title_questions)}")

        titan_rag = create_titan_rag(dim=256)
        chunks = _chunk_context(context)
        print(f"   Chunks: {len(chunks)}")

        chunk_embeddings = embedder.embed_batch(chunks)
        _digest_chunks(embedder, titan_rag, chunks, args.epochs, inline=True)

        is_multihop = dataset_name.lower() in ('hotpotqa',)
        retrievers = create_retrievers(embedder, llm, titan_rag, multihop=is_multihop)

        for name, retriever in retrievers.items():
            print(f"   [{name}] Evaluating...", end="", flush=True)
            accuracy, results = evaluate_retriever(
                retriever, chunks, chunk_embeddings, title_questions,
                verbose=args.verbose, show_progress=not args.verbose,
                embedder=embedder, topk=args.topk
            )

            correct_count = sum(1 for r in results if r["correct"])
            all_results[name]["correct"] += correct_count
            all_results[name]["total"] += len(title_questions)
            all_details[name][title] = results

            print(f" {correct_count}/{len(title_questions)} ({accuracy:.1f}%)")

    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY ({dataset_name})")
    print("=" * 60)
    print(f"Evaluated on {len(selected_titles)} titles")
    print()
    _print_summary_bar(all_results)

    if getattr(args, 'save_results', False):
        _save_results(dataset_name.lower(),
                      {"titles": args.titles, "max_questions": args.max_questions,
                       "epochs": args.epochs, "topk": args.topk},
                      all_results,
                      details=all_details)


def run_squad_mode(args):
    """Run comparison on SQuAD dataset."""
    contexts, questions = load_squad_data()
    run_dataset_mode(args, contexts, questions, "SQuAD")


def run_hotpotqa_mode(args):
    """Run comparison on HotpotQA dataset (multi-hop QA)."""
    contexts, questions = load_hotpotqa_data()
    run_dataset_mode(args, contexts, questions, "HotpotQA")


def run_multidoc_mode(args):
    """Run comparison with multiple articles digested into shared memory."""
    CONTEXTS, TEST_QUESTIONS = load_squad_data()

    print("=" * 60)
    print("BASELINE COMPARISON: Multi-Document Cross-Article Retrieval")
    print(f"Mode: SQuAD Multi-Doc (group_size={args.group_size})")
    print("=" * 60)

    embedder, llm, DEVICE = _init_components()

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

    # Build global chunk pool
    print("\n📄 Building global chunk pool...")
    all_chunks = []
    all_questions = []  # (title, question, expected_answer)

    for title in selected_titles:
        chunks = _chunk_context(CONTEXTS[title])
        all_chunks.extend(chunks)

        questions = TEST_QUESTIONS.get(title, [])
        if questions:
            if len(questions) > args.max_questions:
                questions = questions[:args.max_questions]
            for q, a in questions:
                all_questions.append((title, q, a))

    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Total questions: {len(all_questions)}")

    # Embed all chunks
    print("\n🔢 Embedding all chunks...")
    all_embeddings = embedder.embed_batch(all_chunks)

    # Shared Titan memory
    titan_rag = create_titan_rag(dim=256, device=DEVICE)
    _digest_chunks(embedder, titan_rag, all_chunks, args.epochs, inline=False)

    # Create retrievers
    retrievers = create_retrievers(embedder, llm, titan_rag)

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATING (cross-document retrieval)")
    print("=" * 60)

    eval_questions = [(q, a) for (_, q, a) in all_questions]

    # Per-title tracking
    title_results = {name: {} for name in retrievers}
    for name in retrievers:
        for title in selected_titles:
            title_results[name][title] = []

    results_summary = {}
    multidoc_details = {}  # {retriever: [{title, question, expected, answer, correct, mode}]}
    for name, retriever in retrievers.items():
        print(f"\n📊 Evaluating: {name.upper()}")
        print("-" * 40)
        accuracy, results = evaluate_retriever(
            retriever, all_chunks, all_embeddings, eval_questions,
            verbose=args.verbose, show_progress=not args.verbose,
            embedder=embedder, topk=args.topk
        )
        results_summary[name] = accuracy

        # Annotate results with source title
        annotated = []
        for idx, (title, q, a) in enumerate(all_questions):
            r = dict(results[idx])
            r["title"] = title
            annotated.append(r)
            title_results[name][title].append(results[idx]["correct"])
        multidoc_details[name] = annotated

        print(f"   Overall Accuracy: {accuracy:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    _print_summary_bar(results_summary, show_counts=False)

    # Per-title breakdown
    print("\n" + "-" * 60)
    print("PER-TITLE BREAKDOWN")
    print("-" * 60)
    header = f"  {'Title':<30s}"
    for name in retrievers:
        header += f" {name:>12s}"
    print(header)
    print("  " + "-" * (30 + 13 * len(retrievers)))

    per_title_summary = {}
    for title in selected_titles:
        row = f"  {title[:30]:<30s}"
        per_title_summary[title] = {}
        for name in retrievers:
            bools = title_results[name][title]
            if bools:
                acc = sum(bools) / len(bools) * 100
                row += f" {acc:>10.1f}% "
                per_title_summary[title][name] = round(acc, 1)
            else:
                row += f" {'N/A':>11s} "
        print(row)

    if getattr(args, 'save_results', False):
        _save_results("multidoc",
                      {"group_size": args.group_size, "max_questions": args.max_questions,
                       "epochs": args.epochs, "topk": args.topk,
                       "titles": selected_titles},
                      results_summary,
                      details={"per_title": per_title_summary,
                               "questions": multidoc_details})


def main():
    parser = argparse.ArgumentParser(description="Compare baseline retrieval strategies")

    # Mode selection
    parser.add_argument("--squad", action="store_true",
                       help="Use SQuAD dataset instead of sample essays")
    parser.add_argument("--multi-doc", action="store_true",
                       help="Multi-document mode: digest multiple articles into shared memory")
    parser.add_argument("--hotpotqa", action="store_true",
                       help="Use HotpotQA dataset (multi-hop QA, already multi-document)")

    # Sample essay mode options
    parser.add_argument("--essay", type=str, default="climate",
                       help="Essay to use (climate/ai/space)")

    # SQuAD / HotpotQA mode options
    parser.add_argument("--titles", type=int, default=10,
                       help="Number of titles to evaluate (SQuAD/HotpotQA mode)")
    parser.add_argument("--max-questions", type=int, default=5,
                       help="Max questions per title")

    # Multi-doc mode options
    parser.add_argument("--group-size", type=int, default=5,
                       help="Number of titles to group together (multi-doc mode)")

    # Common options
    parser.add_argument("--epochs", type=int, default=50, help="Digestion epochs")
    parser.add_argument("--topk", type=int, default=0,
                       help="Top-K for retrieval (0=auto based on chunk count)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--save-results", action="store_true",
                       help="Save results to evaluations/ directory as JSON")

    args = parser.parse_args()

    if args.hotpotqa:
        run_hotpotqa_mode(args)
    elif args.multi_doc:
        run_multidoc_mode(args)
    elif args.squad:
        run_squad_mode(args)
    else:
        run_sample_essay_mode(args)


if __name__ == "__main__":
    main()
