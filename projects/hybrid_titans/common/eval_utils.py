"""
Evaluation utilities for answer quality assessment.
Standard SQuAD-style Exact Match (EM) and token-level F1 Score.
"""

import re
import string
from collections import Counter


def normalize_answer(s):
    """Normalize answer string for EM/F1 comparison.
    
    Follows the SQuAD evaluation script convention:
    lowercase → remove articles → remove punctuation → collapse whitespace.
    """
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = s.translate(str.maketrans('', '', string.punctuation))
    # Collapse whitespace
    s = ' '.join(s.split())
    return s


def compute_em(expected, predicted):
    """Exact Match: 1 if normalized strings are identical, else 0."""
    return int(normalize_answer(expected) == normalize_answer(predicted))


def compute_f1(expected, predicted):
    """Token-level F1 Score.
    
    Computes precision and recall over shared tokens between
    normalized expected and predicted answers, returns their
    harmonic mean.
    
    Returns:
        float: F1 score in [0, 1]
    """
    expected_tokens = normalize_answer(expected).split()
    predicted_tokens = normalize_answer(predicted).split()
    
    if not expected_tokens and not predicted_tokens:
        return 1.0
    if not expected_tokens or not predicted_tokens:
        return 0.0
    
    common = Counter(expected_tokens) & Counter(predicted_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(predicted_tokens)
    recall = num_common / len(expected_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
