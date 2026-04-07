# AGENTS.md

Operating guide for AI coding agents working in the TitanRAG repository.

## What this project is

TitanRAG implements the **Titans** architecture (arXiv:2501.00663) — a neural long-term memory that learns at inference time via "surprise"-driven test-time training — and evaluates it as a component of **Retrieval-Augmented Generation** pipelines. The research question is whether neural memory can complement or replace classical retrieval (BM25 + embeddings) on multi-hop QA.

Read `README.md` first for the high-level map. Read `docs/research_roadmap.md` and `docs/Research_Directions.md` to understand *why* a given experiment exists before changing it.

## Repository layout (what lives where)

- `src/main.py` — core Titans: `DeepMemoryModule` and `TitanMAC` / `TitanMAG` / `TitanMAL` variants. `src/main_origin.py` is a reference implementation; do not edit it unless explicitly asked.
- `projects/hybrid_titans/` — the main research surface.
  - `baselines.py` — the four retrieval strategies: `PureRAG`, `TitanOnly`, `HybridRAG`, `AdaptiveHybridRAG`. New retrieval ideas belong here.
  - `compare_baselines.py` — evaluation driver that writes JSON to `evaluations/`.
  - `common/` — shared `embedders.py`, `llm_utils.py`, `eval_utils.py`, `text_utils.py`, `titan_utils.py`. Reuse these; don't fork utilities into experiment files.
- `projects/original_benchmarks/` — toy Titans sanity checks (associative recall). Keep untouched unless reproducing paper claims.
- `data/process_*_data.py` — dataset processors for SQuAD 2.0, HotpotQA, MuSiQue. They download, clean, and serialize to a Python file the rest of the code imports.
- `is_RAG_able/` — query-level classifier (MLP + XGBoost) that predicts whether retrieval helps a given question. Used by `AdaptiveHybridRAG` for soft routing. Trained artifacts live in `is_RAG_able/models/`; reports in `is_RAG_able/results/`.
- `evaluations/` — JSON results organized by dataset (`Squad2.0 Single-Doc`, `Squad2.0 Multi-Doc`, `MuSiQue Multi-Doc`). Multi-doc SQuAD results are reported across seeds 42/52/62 — keep that convention.
- `notebooks/` — generated from `scripts/generate_*_notebook.py`. **Do not hand-edit.** Regenerate after changing source.
- `scripts/` — notebook generators and model downloaders.
- `docs/` — research plans and walkthroughs (some Chinese-language).

## Ground rules for agents

1. **Don't edit generated notebooks.** `notebooks/Baseline_Comparison.ipynb` and `notebooks/Titan_Experiment.ipynb` are produced by `scripts/generate_baseline_notebook.py` and `scripts/generate_notebook_script.py`. Change the source, then regenerate.
2. **Don't touch `src/main_origin.py`** — it's the paper-faithful reference kept for comparison.
3. **Don't commit processed datasets or model weights** beyond what's already tracked. `data/processed_*.py` are regeneratable; `is_RAG_able/models/*.pt|*.pkl` are the exception (already tracked).
4. **Multi-seed discipline.** When adding a Multi-Doc SQuAD result, run all three seeds (42, 52, 62) and write one JSON per seed, matching the existing naming pattern (`multidoc_<method>_seed<NN>.json` or similar). Single-seed multi-doc results are not acceptable.
5. **Reuse `projects/hybrid_titans/common/`.** New embedders, LLM wrappers, tokenization, or eval helpers go there — not inline in an experiment script.
6. **Device agnostic.** Code must run on CPU, CUDA, and Colab TPU. Use the helpers in `common/titan_utils.py` rather than calling `.cuda()` directly.
7. **No silent API/schema changes** to `baselines.py` classes — `compare_baselines.py` and the notebook generators depend on their constructor and `retrieve(...)` signatures.
8. **When adding a new dataset**, follow the `data/process_*_data.py` pattern: download → clean → emit `data/processed_<name>.py` with a single importable list/dict. Add a matching folder under `evaluations/`.

## Running things

```bash
# Data prep (only what you need)
python3 data/process_squad_data.py
python3 data/process_hotpotqa_data.py
python3 data/process_MuSiQue_data.py

# Main evaluation
python3 projects/hybrid_titans/compare_baselines.py

# Regenerate Colab notebooks after editing source
python3 scripts/generate_baseline_notebook.py
python3 scripts/generate_notebook_script.py

# is_RAG_able classifier
python3 is_RAG_able/generate_rag_label_sq.py
python3 is_RAG_able/train_test_model.py
python3 is_RAG_able/train_xgboost.py
```

There is **no test suite** and **no linter config**. "Verification" for a change means: (a) the relevant script runs end-to-end on a small sample, and (b) if you touched a baseline, `compare_baselines.py` still produces a well-formed JSON under `evaluations/`.

## Environment

- Python 3.10, PyTorch, `transformers`, `sentence-transformers`, `accelerate`, `xgboost`, `scikit-learn`.
- FLAN-T5-large is the default LLM. If you see stale-cache errors: `rm -rf ~/.cache/huggingface/hub/models--google--flan-t5-large`.
- Colab is a first-class target — assume a reviewer may run your change on a free-tier GPU or TPU runtime.

## When in doubt

- Researchy change (new baseline, new metric, new dataset) → read `docs/research_roadmap.md` and mirror the structure of the nearest existing example.
- Infra change (utils, device handling, notebook generation) → keep it backward-compatible with `baselines.py` and the notebook generators.
- If a requested change would break the multi-seed reporting convention, the `baselines.py` API, or the generated-notebook workflow, **stop and ask** before proceeding.
