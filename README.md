# TitanRAG

Implementation of **Titans: Learning to Memorize at Test Time** (arXiv:2501.00663).

This project explores the **Titans** architecture, a next-generation neural memory system that learns to memorize historical context *at inference time* (Test-Time Training). We extend this concept to **Retrieval-Augmented Generation (RAG)**, creating a hybrid neuro-symbolic system that "digests" retrieved documents into long-term memory weights.

## 🚀 Key Features

*   **Test-Time Training (TTT)**: The `DeepMemoryModule` updates its weights on-the-fly during inference based on "surprise" (prediction error), allowing it to adaptively memorize new information.
*   **Titans Variants**: Full implementation of all three architectural variants:
    *   **MAC (Memory As Context)**: Prepends retrieved memory to the context window (RAG-style).
    *   **MAG (Memory As Gating)**: Fuses Short-Term Attention and Long-Term Memory via a learned gate.
    *   **MAL (Memory As Layer)**: Processes tokens sequentially through memory before attention.
*   **TitanRAG**: A specialized wrapper for RAG tasks. It "reads" retrieved documents by training on them for a few epochs before answering questions, effectively "baking" knowledge into the neural weights.
*   **Hybrid Retrieval**: Combines **Keyword Search** + **Neural Memory Search** + **Embedding Search** (Ensemble Fusion) for robust context retrieval.
*   **Intelligent Decision Maker**:
    *   **Trainable Decider**: A MLP model that evolves via online AB testing feedback to choose the best strategy.
    *   **Rule-based Decider**: A transparent expert system that makes decisions based on weighted signals (Question Type, Memory Confidence, Retrieval Dispersion).

## 📂 Directory Structure

```
TitanRAG
├── src/                # Core Source Code
│   └── main.py         # Titans Model Implementation (DeepMemoryModule, TitanMAC/MAG/MAL)
├── data/               # Data and Assets
│   └── sample_essays.py # Sample essays for experiments
├── projects/           # Demos and Experiments
│   ├── hybrid_titans/  # Hybrid RAG Experiments (TitanRAG + Flan-T5)
│   └── original_benchmarks/ # Basic functionality tests
├── scripts/            # Utility Scripts
│   ├── generate_notebook_script.py # Generates Colab notebook
│   └── Titan_Colab_Runner.py
├── docs/               # Documentation
│   ├── Research_Directions.md
│   ├── TitanRAG_Research_Plan_CN.md
│   └── Walkthrough_Threshold.md
└── notebooks/          # Processed Notebooks
    └── Titan_Experiment.ipynb
```

## 🛠️ Quick Start (VS Code & Colab)

To run the experiments using the self-contained notebook:

1.  **Setup VS Code**: Install the **Colab** (or Jupyter) extension in VS Code.
2.  **Generate Notebook**: Run the following script to generate the self-contained experiment notebook:
    ```bash
    python scripts/generate_notebook_script.py
    ```
    This will generate `Titan_Experiment.ipynb`.
3.  **Run**: Open `Titan_Experiment.ipynb` in VS Code (or upload to Google Colab) and run all cells. The notebook automates environment setup and experiment execution.
    > **Important**: If you modify any source code locally, you must:
    > 1. Re-run `python scripts/generate_notebook_script.py` to update the notebook.
    > 2. In Colab/VS Code, re-run the **"Setup File System"** cell (or restart the kernel) to propagate changes to the environment.

> **Note**: You can also run locally using the scripts in `projects/`, but you will need a CUDA-ready environment with `deepspeed` installed.

## 🔬 Research & Experiments

*   **[Research Plan (CN)](docs/TitanRAG_Research_Plan_CN.md)**: Detailed roadmap for investigating TitanRAG.
*   **[Walkthrough: Sparse Update](docs/Walkthrough_Threshold.md)**: Analysis of using a "Surprise Threshold" to skip redundant memory updates.
*   **[Research Directions](docs/Research_Directions.md)**: Future ideas including Meta-Optimization and Hierarchical Memory.

## 📄 Reference
Based on the paper:
**Titans: Learning to Memorize at Test Time**
*Google Research*
[https://arxiv.org/pdf/2501.00663v1](https://arxiv.org/pdf/2501.00663v1)
