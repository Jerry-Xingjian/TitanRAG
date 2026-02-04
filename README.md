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
*   **Hybrid Retrieval**: Combines **Keyword Search (BM25)** + **Neural Memory Search** + **Embedding Search** (Ensemble Fusion) for robust context retrieval.
*   **Device Agnostic**: Fully supports **TPU (Google Colab)**, **GPU (CUDA)**, and **CPU** execution with automatic detection.

## 📂 Directory Structure

```
TitanRAG
├── src/                    # Core Titans Architecture
│   └── main.py             # DeepMemoryModule & TitanMAC/MAG/MAL models
├── data/                   # Data Management
│   ├── process_squad_data.py # SQuAD v2.0 downloader & processor
│   ├── processed_squad.py    # Generated SQuAD data file (auto-created)
│   └── sample_essays.py      # Sample essays for quick demos
├── projects/               # Experiments & Implementations
│   └── hybrid_titans/      # Hybrid RAG Implementation
│       ├── baselines.py    # Retrieval strategies (PureRAG, TitanOnly, Hybrid)
│       ├── compare_baselines.py # Main evaluation script
│       ├── common/         # Shared utilities (Embedders, LLM wrappers, TPU helpers)
│       └── ...
├── scripts/                # Utility & Notebook Generators
│   ├── generate_baseline_notebook.py # Generates Baseline_Comparison.ipynb
│   └── generate_notebook_script.py   # Generates Titan_Experiment.ipynb
├── notebooks/              # User Notebooks
│   ├── Baseline_Comparison.ipynb # Main Evaluation Notebook
│   └── Titan_Experiment.ipynb    # Core Feature Demo
└── ...
```

## 🛠️ Data Preparation

The project uses the **SQuAD v2.0** dataset for evaluation.

*   **Automatic**: The scripts (and notebooks) will automatically detect if data is missing and download/process it for you.
*   **Manual**: You can run the processor manually to generate the dataset file:
    ```bash
    python3 data/process_squad_data.py
    ```
    This will:
    1.  Download `train-v2.0.json`.
    2.  Process it into a clean format.
    3.  Save it as `data/processed_squad.py` for easy import.

## 📊 Running Baseline Comparison

Compare three retrieval strategies: **PureRAG** (BM25+Embedding), **TitanOnly** (Memory), and **HybridRAG** (All combined).

### 1. 📓 In Google Colab (Recommended)
This is the easiest way to run the experiments, especially with free **TPU** acceleration.

1.  **Generate the Notebook**:
    Run this local script to package all latest code into a single notebook file:
    ```bash
    python3 scripts/generate_baseline_notebook.py
    ```
    > Output: `notebooks/Baseline_Comparison.ipynb`

2.  **Run in Colab**:
    *   Upload `Baseline_Comparison.ipynb` to Google Colab.
    *   Set Runtime type to **GPU**.
    *   Run all cells. The notebook handles data setup and installation automatically.

### Running locally (Conda / Anaconda)
1. Create a virtual environment: `conda create -n titan-env python=3.10`
2. Activate: `conda activate titan-env`
3. Install dependencies: `conda install pytorch torchvision torchaudio -c pytorch`
4. Finish dependencies: `pip install transformers sentence-transformers accelerate`
5. Remove cache of FlanT5: `rm -rf ~/.cache/huggingface/hub/models--google--flan-t5-large`

## 🔬 Research & Experiments

*   **[Research Plan (CN)](docs/TitanRAG_Research_Plan_CN.md)**: Detailed roadmap for investigating TitanRAG.
*   **[Walkthrough: Sparse Update](docs/Walkthrough_Threshold.md)**: Analysis of "Surprise Threshold" for efficient memory updates.

## 📄 Reference
Based on the paper:
**Titans: Learning to Memorize at Test Time** (Google Research)
[https://arxiv.org/pdf/2501.00663v1](https://arxiv.org/pdf/2501.00663v1)
