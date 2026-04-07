# TitanRAG

Implementation and evaluation of **Titans: Learning to Memorize at Test Time** (arXiv:2501.00663), extended for **Retrieval-Augmented Generation (RAG)**.

TitanRAG explores whether a neural long-term memory that learns *at inference time* can complement — or replace — classical retrieval. We implement the Titans architecture, wrap it for RAG, and benchmark it against strong retrieval baselines on SQuAD 2.0, HotpotQA, and MuSiQue.

## 🚀 Key Features

*   **Test-Time Training (TTT)**: The `DeepMemoryModule` updates its weights on-the-fly during inference based on "surprise" (prediction error), adaptively memorizing new context.
*   **Titans Variants**: Full implementation of all three architectural variants — **MAC** (Memory as Context), **MAG** (Memory as Gating), and **MAL** (Memory as Layer).
*   **Retrieval Baselines**: Four strategies in `projects/hybrid_titans/baselines.py`:
    *   `PureRAG` — BM25 + Embedding fusion
    *   `TitanOnly` — pure neural-memory retrieval
    *   `HybridRAG` — BM25 + Memory + Embedding ensemble
    *   `AdaptiveHybridRAG` — per-chunk soft routing between RAG and Titan signals via a learned scorer
*   **`is_RAG_able` Classifier**: A lightweight MLP / XGBoost model that predicts, per query, whether RAG retrieval is likely to help — used to drive adaptive routing.
*   **Multi-Dataset Evaluation**: SQuAD 2.0 (single- and multi-doc), HotpotQA, and MuSiQue processors included.
*   **Device Agnostic**: TPU (Colab), GPU (CUDA), and CPU, with automatic detection.

## 📂 Directory Structure

```
TitanRAG
├── src/                          # Core Titans architecture
│   ├── main.py                   # DeepMemoryModule + TitanMAC/MAG/MAL
│   └── main_origin.py            # Reference implementation
├── data/                         # Dataset processors
│   ├── process_squad_data.py     # SQuAD v2.0
│   ├── process_hotpotqa_data.py  # HotpotQA
│   ├── process_MuSiQue_data.py   # MuSiQue
│   └── sample_essays.py
├── projects/
│   ├── hybrid_titans/            # Main RAG experiments
│   │   ├── baselines.py          # PureRAG / TitanOnly / Hybrid / AdaptiveHybrid
│   │   ├── compare_baselines.py  # Main evaluation driver
│   │   ├── rag_compare.py
│   │   ├── hybrid_titan_demo.py
│   │   ├── pure_titan_demo.py
│   │   ├── essay_rag_demo.py
│   │   └── common/               # Embedders, LLM wrappers, eval/text/titan utils
│   └── original_benchmarks/      # Toy tasks (associative recall, etc.)
├── is_RAG_able/                  # Query-level "is RAG useful?" classifier
│   ├── generate_rag_label_sq.py  # Label generation on SQuAD
│   ├── train_test_model.py       # MLP trainer
│   ├── train_xgboost.py          # XGBoost trainer
│   ├── rag_infer.py              # Inference-time routing
│   ├── models/                   # Trained weights (.pt, .pkl)
│   └── results/                  # Reports
├── evaluations/                  # Benchmark results (JSON)
│   ├── Squad2.0 Single-Doc/
│   ├── Squad2.0 Multi-Doc/       # AdaptiveRAG / Hybrid / RAGonly / TitansOnly × 3 seeds
│   └── MuSiQue Multi-Doc/
├── notebooks/
│   ├── Baseline_Comparison.ipynb
│   └── Titan_Experiment.ipynb
├── scripts/                      # Notebook generators, model downloaders
└── docs/                         # Research plan, roadmap, walkthroughs
```

## 🛠️ Data Preparation

Dataset processors auto-download and serialize each corpus. Run whichever you need:

```bash
python3 data/process_squad_data.py      # SQuAD v2.0
python3 data/process_hotpotqa_data.py   # HotpotQA
python3 data/process_MuSiQue_data.py    # MuSiQue
```

The evaluation scripts and notebooks will trigger these automatically if the processed files are missing.

## 📊 Running Baseline Comparison

Compare the four retrieval strategies on a dataset of your choice:

```bash
python3 projects/hybrid_titans/compare_baselines.py
```

Results are written as JSON under `evaluations/<dataset>/`. The Multi-Doc SQuAD 2.0 results are reported across three seeds (42 / 52 / 62) for each baseline.

### In Google Colab

1. Generate the packaged notebook:
   ```bash
   python3 scripts/generate_baseline_notebook.py
   ```
   Output: `notebooks/Baseline_Comparison.ipynb`
2. Upload to Colab, set runtime to **GPU** (or **TPU**), and run all cells.

### Running locally (Conda)

```bash
conda create -n titan-env python=3.10 && conda activate titan-env
conda install pytorch torchvision torchaudio -c pytorch
pip install transformers sentence-transformers accelerate xgboost scikit-learn
```

If you hit a stale FLAN-T5 cache:
```bash
rm -rf ~/.cache/huggingface/hub/models--google--flan-t5-large
```

## 🧠 `is_RAG_able`: Query-Level RAG Routing

Not every question benefits from retrieval — sometimes the LLM already knows the answer and retrieval only adds noise. `is_RAG_able/` trains a classifier on SQuAD-derived labels that predicts, per question, whether RAG is likely to help. This signal feeds `AdaptiveHybridRAG` for per-chunk soft routing between retrieved context and Titan memory. See `is_RAG_able/判别模型训练流程说明.md` for the training pipeline.

## 🔬 Research & Docs

*   [Research Roadmap](docs/research_roadmap.md)
*   [Research Directions](docs/Research_Directions.md)
*   [Research Plan (CN)](docs/TitanRAG_Research_Plan_CN.md)
*   [Walkthrough: Surprise Threshold (CN)](docs/Walkthrough_Threshold_CN.md)

## 📄 Reference

**Titans: Learning to Memorize at Test Time** — Google Research
<https://arxiv.org/pdf/2501.00663v1>
