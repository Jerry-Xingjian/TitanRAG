
import base64
import json
import os

def get_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Read files
# Assuming script is run from root or we adjust path
import sys
try:
    main_b64 = get_b64("src/main.py")
except FileNotFoundError:
    # Fallback if running from scripts/ directory
    main_b64 = get_b64("../src/main.py")

def get_b64_safe(path):
    if os.path.exists(path):
        return get_b64(path)
    elif os.path.exists("../" + path):
        return get_b64("../" + path)
    else:
        raise FileNotFoundError(f"Could not find {path}")

ar_b64 = get_b64_safe("projects/original_benchmarks/associative_recall.py")
train_b64 = get_b64_safe("projects/original_benchmarks/train_toy.py")
rag_b64 = get_b64_safe("projects/hybrid_titans/rag_compare.py")
essays_b64 = get_b64_safe("data/sample_essays.py")
essay_demo_b64 = get_b64_safe("projects/hybrid_titans/essay_rag_demo.py")
hybrid_titan_b64 = get_b64_safe("projects/hybrid_titans/hybrid_titan_demo.py")

# Define Notebook Cells
cells = []

# Cell 1: Intro
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Titan Experiment Runner\n",
        "This notebook sets up the environment and runs the Associative Recall experiment on the Titans model."
    ]
})

# Cell 2: Install
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. Install Dependencies"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install dependencies\n",
        "!pip install torch deepspeed transformers tqdm einops sentence-transformers"
    ]
})

# Cell 3: Setup File System
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2. Setup File System"]
})
setup_code = [
    "import os\n",
    "import base64\n",
    "\n",
    "print(f'Current working directory: {os.getcwd()}')\n",
    "\n",
    "def write_file(path, b64_content):\n",
    "    target_path = os.path.abspath(path)\n",
    "    directory = os.path.dirname(target_path)\n",
    "    if directory:\n",
    "        os.makedirs(directory, exist_ok=True)\n",
    "    with open(target_path, \"wb\") as f:\n",
    "        f.write(base64.b64decode(b64_content))\n",
    "    print(f\"Created {target_path}\")\n",
    "\n",
    f"files = {{\n",
    f"    'src/main.py': '{main_b64}',\n",
    f"    'projects/original_benchmarks/associative_recall.py': '{ar_b64}',\n",
    f"    'projects/original_benchmarks/train_toy.py': '{train_b64}',\n",
    f"    'projects/hybrid_titans/rag_compare.py': '{rag_b64}',\n",
    f"    'data/sample_essays.py': '{essays_b64}',\n",
    f"    'projects/hybrid_titans/essay_rag_demo.py': '{essay_demo_b64}',\n",
    f"    'projects/hybrid_titans/hybrid_titan_demo.py': '{hybrid_titan_b64}'\n",
    "}\n",
    "\n",
    "for path, content in files.items():\n",
    "    write_file(path, content)\n",
    "\n",
    "print('Files in current directory:', os.listdir('.'))\n",
    "if os.path.exists('experiments'):\n",
    "    print('Files in experiments:', os.listdir('experiments'))\n"
]
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": setup_code
})

# Cell 4: Run Baseline (Dense)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 3. Run Baseline (Dense Update)"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run Baseline (Dense Update)\n",
        "# Decreasing vocab_size to 16 to ensure model can learn quickly\n",
        "import os\n",
        "if os.path.exists('projects/original_benchmarks/train_toy.py'):\n",
        "    print('Running Baseline (Dense Update, Vocab=16)...')\n",
        "    !python projects/original_benchmarks/train_toy.py --model_variant MAC --num_epochs 5 --batch_size 16 --vocab_size 16 --threshold 0.0\n",
        "else:\n",
        "    print('ERROR: projects/original_benchmarks/train_toy.py not found!')\n"
    ]
})

# Cell 5: Run Innovation (Sparse)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 4. Run Innovation (Sparse Update)"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run Innovation (Sparse Update)\n",
        "print('Running Sparse Update (Threshold = 0.5, Vocab=16)...')\n",
        "!python projects/original_benchmarks/train_toy.py --model_variant MAC --num_epochs 5 --batch_size 16 --vocab_size 16 --threshold 0.5\n"
    ]
})

# Cell 6: TitanRAG Demo
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 5. TitanRAG Prototype Demo"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# TitanRAG Prototype Demo\n",
        "# Demonstrating 'Retrieve-then-Digest' mechanism\n",
        "import torch\n",
        "import sys, os\n",
        "sys.path.append(os.path.abspath('src'))\n",
        "from main import TitanMAC, TitanRAG\n",
        "\n",
        "print('Initializing TitanRAG...')\n",
        "# 1. Initialize Base Model\n",
        "base_model = TitanMAC(dim=128, chunk_size=32, hidden_dim=128, memory_depth=2, num_persistent_tokens=4, threshold=0.1)\n",
        "rag_model = TitanRAG(base_model)\n",
        "\n",
        "# 2. Simulate Data (Embeddings)\n",
        "print('Simulating Retrieval...')\n",
        "doc1 = torch.randn(1, 64, 128) # Retrieved Doc A\n",
        "doc2 = torch.randn(1, 64, 128) # Retrieved Doc B\n",
        "query = torch.randn(1, 16, 128) # User Query\n",
        "\n",
        "# 3. Run Pipeline\n",
        "print('Executing Digestion & Inference...')\n",
        "output = rag_model.query_with_context(query, [doc1, doc2])\n",
        "\n",
        "print('Success! Output shape:', output.shape)\n",
        "print('The model weights have been updated on-the-fly with document info.')\n"
    ]
})

# Cell 7: RAG Comparison Experiment (Small Scale)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 6. RAG Comparison Experiment (Small Scale)"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Experiment: RAG Comparison - Small Scale (Within Context Window)\n",
        "# Both methods should succeed.\n",
        "print('--- Running RAG Comparison (Small Scale: 10 Docs) ---')\n",
        "!python projects/hybrid_titans/rag_compare.py --num_docs 10 --doc_len 32\n"
    ]
})

# Cell 8: RAG Comparison Experiment (Large Scale)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 7. RAG Comparison Experiment (Large Scale)"]
})
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Experiment: RAG Comparison - Large Scale (Exceeds Context Window)\n",
        "# Traditional RAG should FAIL (Context Limit).\n",
        "# TitanRAG should PASS (Infinite Memory).\n",
        "print('\\n--- Running RAG Comparison (Large Scale: 100 Docs) ---')\n",
        "!python projects/hybrid_titans/rag_compare.py --num_docs 100 --doc_len 32\n"
    ]
})

# Cell 9: Essay RAG Demo
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 8. Essay-based TitanRAG Demo\n",
        "This cell demonstrates TitanRAG on real text: Climate Essay.\n",
        "The model digests the essay and then answers verification questions."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run Essay RAG Demo (Climate Essay - Random Embedder Baseline)\n",
        "print('--- Essay RAG Demo: Climate (Random Embedder) ---')\n",
        "!python projects/hybrid_titans/essay_rag_demo.py --essay climate --epochs 50 --embedder random\n"
    ]
})

# Cell 10: Semantic Embedder - All Essays
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 9. Semantic Embedder (sentence-transformers)\n",
        "Now running with **semantic embeddings** for meaningful text similarity.\n",
        "This should show significant improvement in Hybrid RAG performance!"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run ALL Essays with Semantic Embedder\n",
        "print('=' * 60)\n",
        "print('SEMANTIC EMBEDDER TEST - ALL ESSAYS')\n",
        "print('=' * 60)\n",
        "\n",
        "# Climate Essay\n",
        "print('\\n--- Essay 1/3: Climate Change ---')\n",
        "!python projects/hybrid_titans/essay_rag_demo.py --essay climate --epochs 50 --embedder semantic\n",
        "\n",
        "# AI Essay\n",
        "print('\\n--- Essay 2/3: Artificial Intelligence ---')\n",
        "!python projects/hybrid_titans/essay_rag_demo.py --essay ai --epochs 50 --embedder semantic\n",
        "\n",
        "# Space Essay\n",
        "print('\\n--- Essay 3/3: Solar System ---')\n",
        "!python projects/hybrid_titans/essay_rag_demo.py --essay space --epochs 50 --embedder semantic\n"
    ]
})

# Cell 11: Hybrid Titan Demo (Direction A)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 10. Hybrid Titan: TitanRAG + Flan-T5 + Decision Maker\n",
        "This cell demonstrates **Direction A** with intelligent mode selection:\n",
        "- **Decision Maker** automatically chooses between titans_only and hybrid modes\n",
        "- TitanRAG performs retrieval using Ensemble Fusion (Keyword + Memory + Embedding)\n",
        "- Flan-T5 generates natural language answers from context\n",
        "- **Online Learning**: Model learns from each Q&A pair\n",
        "\n",
        "**Decision Signals**:\n",
        "1. Question Type (30%): Factual vs Reasoning\n",
        "2. Memory Confidence (40%): LTM output strength\n",
        "3. Retrieval Dispersion (30%): Score distribution"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Hybrid Titan Demo: TitanRAG Retrieval + Flan-T5 Generation + Decision Maker\n",
        "# Running ALL THREE ESSAYS with Online Learning\n",
        "\n",
        "print('=' * 60)\n",
        "print('HYBRID TITAN DEMO - ALL ESSAYS')\n",
        "print('=' * 60)\n",
        "\n",
        "# 1. Climate Essay (Default Decision Maker Settings)\n",
        "print('\\n' + '─' * 60)\n",
        "print('📖 Essay 1/3: CLIMATE CHANGE (Default Settings)')\n",
        "print('─' * 60)\n",
        "!python projects/hybrid_titans/hybrid_titan_demo.py --essay climate --epochs 50 --topk 5\n",
        "\n",
        "# 2. AI Essay (Custom: More aggressive titans_only)\n",
        "print('\\n' + '─' * 60)\n",
        "print('📖 Essay 2/3: ARTIFICIAL INTELLIGENCE (High Threshold)')\n",
        "print('─' * 60)\n",
        "!python projects/hybrid_titans/hybrid_titan_demo.py --essay ai --epochs 50 --topk 5 --decision_threshold 0.6\n",
        "\n",
        "# 3. Space Essay (Custom: Emphasize question type)\n",
        "print('\\n' + '─' * 60)\n",
        "print('📖 Essay 3/3: SOLAR SYSTEM (Question-Type Weighted)')\n",
        "print('─' * 60)\n",
        "!python projects/hybrid_titans/hybrid_titan_demo.py --essay space --epochs 50 --topk 5 --question_weight 0.5 --memory_weight 0.3 --dispersion_weight 0.2\n"
    ]
})

# Notebook Structure
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_dir = "notebooks"
output_file = "Titan_Experiment.ipynb"
output_path = os.path.join(output_dir, output_file)

# Ensure output directory exists (relative to where script is run)
if not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
    except OSError:
        pass # Might be running inside scripts dir, handled below or ignore

# Handle path if running from inside scripts/
if not os.path.exists(output_dir) and os.path.exists("../notebooks"):
    output_path = os.path.join("../notebooks", output_file)

# Delete existing file if it exists
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"Deleted existing {output_path}")

# Also check for one in current dir and delete to avoid confusion
if os.path.exists(output_file) and output_file != output_path:
    os.remove(output_file)
    print(f"Deleted existing {output_file} from current directory")

with open(output_path, "w") as f:
    json.dump(notebook, f, indent=2)

print(f"{output_file} generated successfully at {output_path}")
