# Research Plan: Optimizing Titans for Next-Gen Memory

## Phase 1: Foundation & Baseline (Completed)
*   **Objective**: Reproduce the official Titans implementation and establish a quantitative baseline.
*   **Key Actions**:
    *   [x] Analyze `DeepMemoryModule` and `TitanMAC/MAG/MAL` architectures.
    *   [x] Implement an **Associative Recall** toy task to measure memory capabilities.
    *   [x] Establish baseline performance (Accuracy ~39.6%, Loss ~1.62) on a small-scale setup.

## Phase 2: Core Research Directions (Proposed)

### Direction 1: Adaptive Sparse Memory (Current Focus)
*   **Concept**: **"Memorize Only What Matters."**
*   **Problem**: Updating memory for every token is computationally expensive and introduces noise.
*   **Innovation**: Introduce a **Surprise Threshold ($\tau$)**. The memory module only updates its weights when the prediction error (surprise) exceeds $\tau$.
*   **Status**: **Proof-of-Concept Successful.**
    *   Achieved **40.04% Accuracy** (vs Baseline 39.61%) with **~4% Speedup** using $\tau=0.5$.
*   **Next Steps**:
    *   Scale up to larger datasets (e.g., WikiText-103).
    *   Visualize the "Surprise Distribution" to understand *what* the model chooses to memorize.

### Direction 2: Meta-Optimized Memory Update
*   **Concept**: **"Learn How to Memorize."**
*   **Problem**: Fixed scalar hyperparameters ($\alpha, \theta, \eta$) are suboptimal for diverse neural dynamics.
*   **Innovation**: Parameterize the update rules.
    *   Use **vectorized** or **matrix-based** learning rates instead of scalars.
    *   Employ **Meta-Learning** to let the model learn its own optimal update dynamics during pre-training.
*   **Expected Outcome**: Faster convergence on few-shot adaptation tasks.

### Direction 3: Hierarchical Multi-Scale Memory
*   **Concept**: **"Short-Term vs. Long-Term Memory."**
*   **Problem**: A single memory module struggles to balance capturing recent details and retaining global context.
*   **Innovation**: Design a **Two-Tier Memory System**.
    *   **Fast Memory**: High update rate, high decay (Working Memory).
    *   **Slow Memory**: Low update rate, low decay (Long-Term Storage).
*   **Expected Outcome**: Superior performance on infinite-context tasks (e.g., >100k tokens).

### Direction 4: Hybrid Memory (Neural + Symbolic)
*   **Concept**: **"Combining Fuzzy Intuition with Exact Recall."**
*   **Problem**: Neural memory (Titans) is lossy and struggles with exact string reproduction (e.g., phone numbers, code snippets), while KV Cache/RAG is exact but expensive/limited.
*   **Innovation**: creating a **Neuro-Symbolic Architecture**.
    *   **Titan + KV Cache**: Use Titans as a "Garbage Collector" for the KV Cache. When tokens are evicted from the sliding window, train them into the LTM instead of discarding them ("Evaluation is not eviction, but consolidation").
    *   **Titan + RAG**: Retrieve documents using RAG, then "digest" them into Titans weights via Test-Time Training, creating a "Knowledge-Infused Model" with a clean context window.
    *   **Results (Experiment `rag_compare.py`)**:
        *   **Pure Memory**: All architectures (MAC, MAG, MAL) demonstrated consistent ~80% fidelity, proving `DeepMemoryModule` is a robust, transferable component.
        *   **Hybrid RAG (Context + Memory)**:
            *   **TitanMAG (Gated)**: **93.5%+ Fidelity**. Best performance. Gating mechanism naturally preserves signal integrity.
            *   **TitanMAC (Concat)**: **97.3%+ Fidelity**. Excellent performance (after Identity Pre-training). Concatenation allows Attention to selectively copy.
            *   **TitanMAL (Residual)**: **~30% Fidelity**. Failed due to additive noise interference. Proves that Gating/Concatenation is required for robust Hybrid Memory.
    *   **Insight**: "Identity Pre-training" is crucial. The model must explicitly learn to copy from context (Attention) to complement its weight-based memory (LTM).
*   **Expected Outcome**: Best of both worlds—exact recall for recent/critical details layer, and infinite semantic context from the neural memory.