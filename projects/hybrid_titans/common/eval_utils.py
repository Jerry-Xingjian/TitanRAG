"""
Evaluation utilities for retrieval and answer quality assessment.
"""

import re
import math


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Answer quality evaluation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dynamic top-k computation
# ---------------------------------------------------------------------------

def compute_topk(num_chunks):
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


# ---------------------------------------------------------------------------
# Retriever evaluation loop
# ---------------------------------------------------------------------------

def evaluate_retriever(retriever, documents, doc_embeddings, questions,
                       verbose=True, show_progress=False, embedder=None,
                       topk=None):
    """Evaluate a retriever on a set of questions.

    Args:
        retriever: BaseRetriever instance
        documents: List of document chunks
        doc_embeddings: Pre-computed embeddings for chunks
        questions: List of (question, expected_answer) tuples
        verbose: Print per-question details
        show_progress: Show compact progress dots
        embedder: Embedder for semantic similarity fallback
        topk: Number of top chunks to retrieve (0 or None = auto)

    Returns:
        tuple: (accuracy_percentage, list_of_result_dicts)
    """
    correct = 0
    total = len(questions)

    # Dynamic topk if not specified
    if topk is None or topk <= 0:
        topk = compute_topk(len(documents))

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
