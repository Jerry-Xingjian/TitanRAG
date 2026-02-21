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
from typing import List, Dict, Tuple
import torch
import sys
import os
import re
import random
import io
import json
import math
from datetime import datetime

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
from baselines import create_retrievers, BaseRetriever, HybridRAGV2

# Add try-except for rag_infer to fail gracefully if skipped
try:
    from is_RAG_able.rag_infer import RAGConfidenceScorer
except ImportError:
    RAGConfidenceScorer = None
    print("Warning: RAGConfidenceScorer not found. Hybrid_v2 may not work.")


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


def evaluate_answer_quality(expected, got, embedder=None):
    """Flexible answer evaluation using multiple criteria.
    
    Args:
        expected: Expected answer string
        got: Model-generated answer string
        embedder: Optional SentenceTransformerEmbedder for semantic similarity fallback
    """
    expected_lower = expected.lower().strip()
    got_lower = got.lower().strip()
    
    # 0. Yes/No shortcut (important for HotpotQA comparison questions)
    if expected_lower in ('yes', 'no'):
        got_first = got_lower.split()[0] if got_lower else ''
        if got_first.rstrip('.,!') == expected_lower:
            return True
        if expected_lower in got_lower and expected_lower != 'no':
            return True
        if expected_lower == 'no' and ('no,' in got_lower or 'no.' in got_lower or got_lower == 'no'):
            return True
        return False
    
    # 0.5 Normalize articles and common prefixes
    def strip_articles(s):
        for prefix in ('the ', 'a ', 'an '):
            if s.startswith(prefix):
                s = s[len(prefix):]
        return s.strip()
    
    exp_norm = strip_articles(expected_lower)
    got_norm = strip_articles(got_lower)
    
    # 1. Bidirectional containment (with and without articles)
    if expected_lower in got_lower or got_lower in expected_lower:
        return True
    if exp_norm in got_norm or got_norm in exp_norm:
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
    
    # 4. Keyword overlap (adaptive threshold based on answer length)
    expected_keywords = extract_keywords(expected)
    got_keywords = extract_keywords(got)
    if expected_keywords:
        overlap = len(expected_keywords & got_keywords)
        threshold = 0.3 if len(expected_keywords) <= 3 else 0.4
        if overlap / len(expected_keywords) >= threshold:
            return True
    
    # 5. Semantic similarity fallback (embedding-based)
    if embedder is not None:
        try:
            import torch.nn.functional as F
            exp_emb = embedder(expected).reshape(-1, embedder.target_dim).mean(dim=0)
            got_emb = embedder(got).reshape(-1, embedder.target_dim).mean(dim=0)
            sim = F.cosine_similarity(exp_emb.unsqueeze(0), got_emb.unsqueeze(0)).item()
            if sim >= 0.75:
                return True
        except Exception:
            pass
    
    return False


def _compute_topk(num_chunks):
    """Dynamically compute topk based on chunk pool size.
    
    - ≤10 chunks: topk=3 (small docs, most chunks relevant)
    - 10-50 chunks: topk=3-5 (moderate, need selectivity)
    - 50-500 chunks: topk=5-6 (large pool, need more coverage)
    - 500+ chunks: topk=7 (very large, max coverage)
    """
    if num_chunks <= 10:
        return 3
    topk = 3 + int(math.log2(num_chunks / 10))
    return max(3, min(topk, 7))


def evaluate_retriever(retriever, documents, doc_embeddings, questions,
                       verbose=True, show_progress=False, embedder=None,
                       topk=None):
    """Evaluate a retriever on a set of questions."""
    correct = 0
    total = len(questions)
    
    # Dynamic topk if not specified
    if topk is None or topk <= 0:
        topk = _compute_topk(len(documents))
    
    results = []
    progress_chars = []
    
    for i, (question, expected) in enumerate(questions):
        result = retriever.answer(question, documents, doc_embeddings, topk=topk)
        is_correct = evaluate_answer_quality(expected, result["answer"], embedder=embedder)
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
    
    # Init Scorer
    scorer = None
    if RAGConfidenceScorer:
        scorer = RAGConfidenceScorer(device=DEVICE)
        
    # --- PREPARE DATA FOR V2 (Selective Digestion) ---
    v2_ready = False
    if scorer and "hybrid_v2" in retrievers:
        try:
            # 1. Score chunks
            q_dummy = [questions[0][0]] * len(chunks) if questions else [""] * len(chunks)
            scores = scorer.predict_batch(q_dummy, chunks)
            
            # 2. Split chunks
            rag_chunks, rag_embs = [], []
            titan_chunks, titan_embs = [], []
            
            for idx, (chunk, score) in enumerate(zip(chunks, scores)):
                if score >= 0.5:
                    rag_chunks.append(chunk)
                    rag_embs.append(chunk_embeddings[idx])
                else:
                    titan_chunks.append(chunk)
                    titan_embs.append(chunk_embeddings[idx])
                    
            rag_embeddings_t = torch.stack(rag_embs) if rag_embs else None
            titan_embeddings_t = torch.stack(titan_embs) if titan_embs else None
            
            print(f"   [V2 Split] Retrievable: {len(rag_chunks)} (RAG), Hard: {len(titan_chunks)} (Titan)")
            
            # 3. Dedicated Titan memory for hard chunks
            titan_v2 = create_titan_rag(dim=256)
            if titan_chunks:
                print(f"\n🧠 Digesting {len(titan_chunks)} hard chunks into V2 memory ({args.epochs} epochs)...")
                _digest_chunks(embedder, titan_v2, titan_chunks, args.epochs, inline=True)
            
            # 4. Attach specific memory to v2
            retrievers["hybrid_v2"].titan_rag = titan_v2
            v2_ready = True
        except Exception as e:
            print(f"   [V2 Error] Failed to prepare split retrieval: {e}")

    print("\n" + "=" * 60 + "\n")
    
    results_summary = {}
    all_details = {}
    for name, retriever in retrievers.items():
        print(f"📊 Evaluating: {name.upper()}")
        print("-" * 40)
        
        if name == "hybrid_v2":
            if not v2_ready:
                print("   Skipped due to V2 setup failure.")
                continue
            accuracy, results = evaluate_retriever_v2(
                retriever, 
                rag_chunks, rag_embeddings_t,
                titan_chunks, titan_embeddings_t,
                questions,
                verbose=True, show_progress=False,
                embedder=embedder, topk=args.topk
            )
        else:
            accuracy, results = evaluate_retriever(
                retriever, chunks, chunk_embeddings, questions,
                verbose=True, show_progress=False,
                embedder=embedder, topk=args.topk
            )
            
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
                      {"essay": args.essay, "epochs": args.epochs},
                      results_summary,
                      details=all_details)


def _init_components():
    """Initialize shared components (embedder, LLM). Returns (embedder, llm, DEVICE)."""
    print("\n📦 Loading components...")
    from common.embedders import DEVICE
    embedder = SentenceTransformerEmbedder(target_dim=256, device=DEVICE)
    llm = FlanT5Generator("google/flan-t5-xl", device=DEVICE)
    return embedder, llm, DEVICE


def _chunk_context(context):
    """Split a context string into chunks, respecting document boundaries.
    
    For multi-document contexts (HotpotQA style with '# Title' headers),
    chunks are created within each document to avoid mixing content
    from different source documents in the same chunk.
    """
    # Check if context has multiple document sections (e.g. HotpotQA)
    import re
    doc_sections = re.split(r'\n(?=# )', context)
    
    if len(doc_sections) > 1:
        # Multi-document: chunk each section independently
        all_chunks = []
        for section in doc_sections:
            section = section.strip()
            if not section:
                continue
            section_chunks = split_into_chunks(section, sentences_per_chunk=3, overlap_sentences=1)
            if section_chunks:
                all_chunks.extend(section_chunks)
            elif len(section) >= 30:
                all_chunks.append(section)
        if all_chunks:
            return all_chunks
    
    # Single-document or fallback
    chunks = split_into_chunks(context, sentences_per_chunk=2, overlap_sentences=1)
    if not chunks:
        chunks = [p.strip() for p in context.split('\n\n') if p.strip()]
    if not chunks:
        chunks = [context]
    return chunks


def _digest_chunks(embedder, titan_rag, chunks, epochs, inline=False):
    """Digest chunks into Titan memory.
    
    Args:
        inline: If True, print progress inline ("10 20 30 done").
                If False, print epoch lines.
    """
    digest_epochs = min(epochs, 500)
    if inline:
        print(f"   Digesting ({digest_epochs} epochs): ", end="", flush=True)
    else:
        print(f"\n🧠 Digesting {len(chunks)} chunks into shared memory ({digest_epochs} epochs)...")

    # Pre-compute embeddings once (chunks don't change between epochs)
    cached_emb = embedder.embed_batch(chunks).unsqueeze(0)

    for epoch in range(digest_epochs):
        titan_rag.digest_knowledge(cached_emb)
        if (epoch + 1) % 10 == 0:
            if inline:
                print(f"{epoch+1}", end=" ", flush=True)
            else:
                print(f"   Epoch {epoch + 1}/{digest_epochs}")

    if inline:
        print("done")


def _print_summary_bar(results_dict, show_counts=True):
    """Print a bar-chart summary of retriever results."""
    for name, stats in results_dict.items():
        if isinstance(stats, dict) and stats.get("total", 0) > 0:
            accuracy = stats["correct"] / stats["total"] * 100
            bar = "█" * int(accuracy / 10) + "░" * (10 - int(accuracy / 10))
            if show_counts:
                print(f"  {name:12s}: {bar} {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
            else:
                print(f"  {name:12s}: {bar} {accuracy:.1f}%")
        elif isinstance(stats, (int, float)):
            accuracy = stats
            bar = "█" * int(accuracy / 10) + "░" * (10 - int(accuracy / 10))
            print(f"  {name:12s}: {bar} {accuracy:.1f}%")


def _save_results(dataset_name, config, summary, details=None):
    """Save evaluation results to evaluations/ directory as JSON.

    Args:
        dataset_name: e.g. 'hotpotqa', 'squad', 'multidoc', 'essay_climate'
        config: dict of run configuration (epochs, topk, titles, etc.)
        summary: dict of {retriever_name: accuracy_or_stats}
        details: optional list of per-question result dicts
    """
    eval_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'evaluations')
    os.makedirs(eval_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{dataset_name}_{timestamp}.json"

    # Show relative path for portability
    try:
        rel_path = os.path.relpath(os.path.join(eval_dir, filename))
    except ValueError:
        rel_path = os.path.join(eval_dir, filename)

    # Normalize summary to {name: {accuracy, correct, total}}
    norm_summary = {}
    for name, stats in summary.items():
        if isinstance(stats, dict) and "total" in stats:
            total = stats["total"]
            correct = stats["correct"]
            norm_summary[name] = {
                "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
                "correct": correct, "total": total
            }
        elif isinstance(stats, (int, float)):
            norm_summary[name] = {"accuracy": round(stats, 1)}

    data = {
        "dataset": dataset_name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "summary": norm_summary,
    }
    if details:
        data["details"] = details

    filepath = os.path.join(eval_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: {rel_path}")


def evaluate_retriever_v2(retriever: HybridRAGV2, 
                          rag_chunks: List[str], rag_embeddings: torch.Tensor,
                          titan_chunks: List[str], titan_embeddings: torch.Tensor,
                          questions: List[Tuple[str, str]], verbose: bool = False, 
                          show_progress: bool = True, embedder=None, topk: int = 5) -> Tuple[float, List[Dict]]:
    """Evaluate HybridRAGV2 using split document pools."""
    correct = 0
    total = len(questions)
    results = []
    progress_chars = []
    
    for i, (question, expected) in enumerate(questions):
        try:
            ans_data = retriever.answer_split(
                question, 
                rag_chunks, rag_embeddings,
                titan_chunks, titan_embeddings,
                topk=topk
            )
            model_answer = ans_data["answer"]
            is_correct = evaluate_answer_quality(expected, model_answer, embedder=embedder)
            
            if is_correct:
                correct += 1
                
            results.append({
                "question": question,
                "expected": expected,
                "answer": model_answer,
                "correct": is_correct,
                "mode": "split_topk"
            })
            
            if show_progress:
                status_char = "." if is_correct else "x"
                progress_chars.append(status_char)
                if (i + 1) % 10 == 0 or i == total - 1:
                    print(f"      [{i+1}/{total}] {''.join(progress_chars[-10:])}", end="\\r")
            
            if verbose:
                status = "ok" if is_correct else "X"
                print(f"      [{i+1}/{total}] {status} Q: {question[:50]}")
                print(f"           Exp: {expected[:40]}")
                print(f"           Got: {model_answer[:40]}")
                
        except Exception as e:
            if verbose:
                print(f"      [{i+1}/{total}] Error Q: {question[:50]} -> {e}")
            results.append({
                "question": question,
                "expected": expected,
                "answer": f"ERROR: {str(e)}",
                "correct": False,
                "mode": "error"
            })
            
    if show_progress:
        print()
            
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy, results


def run_dataset_mode(args, contexts, questions, dataset_name):
    """
    Shared evaluation logic for dataset-based modes (SQuAD, HotpotQA, etc.).
    
    Each title gets its own Titan memory, chunks are digested independently,
    and all retrievers are evaluated per-title.
    """
    print("=" * 60)
    print("BASELINE COMPARISON: PureRAG vs TitanOnly vs HybridRAG vs HybridRAGV2")
    print(f"Mode: {dataset_name} ({args.titles} titles)")
    print("=" * 60)

    embedder, llm, DEVICE = _init_components()
    
    # Init Scorer
    scorer = None
    if RAGConfidenceScorer:
        scorer = RAGConfidenceScorer(device=DEVICE)

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
        "hybrid": {"correct": 0, "total": 0},
        "hybrid_v2": {"correct": 0, "total": 0}
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

        chunks = _chunk_context(context)
        print(f"   Chunks: {len(chunks)}")
        
        # --- PREPARE DATA FOR V1 (All chunks digested) ---
        chunk_embeddings = embedder.embed_batch(chunks)
        
        titan_rag = create_titan_rag(dim=256)
        _digest_chunks(embedder, titan_rag, chunks, args.epochs, inline=True)

        is_multihop = dataset_name.lower() in ('hotpotqa',)
        retrievers = create_retrievers(embedder, llm, titan_rag, multihop=is_multihop)

        # --- PREPARE DATA FOR V2 (Selective Digestion) ---
        v2_ready = False
        if scorer and "hybrid_v2" in retrievers:
            try:
                # 1. Score chunks
                q_dummy = [title_questions[0][0]] * len(chunks) if title_questions else [""] * len(chunks)
                scores = scorer.predict_batch(q_dummy, chunks)
                
                # 2. Split chunks (threshold 0.5)
                rag_chunks, rag_embs = [], []
                titan_chunks, titan_embs = [], []
                
                for idx, (chunk, score) in enumerate(zip(chunks, scores)):
                    if score >= 0.5:
                        rag_chunks.append(chunk)
                        rag_embs.append(chunk_embeddings[idx])
                    else:
                        titan_chunks.append(chunk)
                        titan_embs.append(chunk_embeddings[idx])
                        
                rag_embeddings_t = torch.stack(rag_embs) if rag_embs else None
                titan_embeddings_t = torch.stack(titan_embs) if titan_embs else None
                
                print(f"   [V2 Split] Retrievable: {len(rag_chunks)} (RAG), Hard: {len(titan_chunks)} (Titan)")
                
                # 3. Dedicated Titan memory for hard chunks
                titan_v2 = create_titan_rag(dim=256)
                if titan_chunks:
                    _digest_chunks(embedder, titan_v2, titan_chunks, args.epochs, inline=True)
                
                # 4. Attach specific memory to v2
                retrievers["hybrid_v2"].titan_rag = titan_v2
                v2_ready = True
            except Exception as e:
                print(f"   [V2 Error] Failed to prepare split retrieval: {e}")

        # --- EVALUATE ALL ---
        for name, retriever in retrievers.items():
            if name == "hybrid_v2":
                if not v2_ready:
                    continue
                print(f"   [{name}] Evaluating...", end="", flush=True)
                accuracy, results = evaluate_retriever_v2(
                    retriever, 
                    rag_chunks, rag_embeddings_t,
                    titan_chunks, titan_embeddings_t,
                    title_questions,
                    verbose=args.verbose, show_progress=not args.verbose,
                    embedder=embedder, topk=args.topk
                )
            else:
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
