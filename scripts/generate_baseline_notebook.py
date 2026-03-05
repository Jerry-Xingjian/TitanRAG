#!/usr/bin/env python3
"""
Generate Baseline Comparison Notebook.

This script generates a Jupyter notebook that compares the three
baseline retrieval strategies: PureRAG, TitanOnly, and HybridRAG.

Structure similar to Titan_Experiment.ipynb:
1. Install Dependencies
2. Setup File System (creates all files in one cell)
3. Run Baseline Comparison (sample essays, SQuAD, HotpotQA, or MuSiQue)

Usage:
    python3 generate_baseline_notebook.py
"""

import base64
import json
import os


def get_b64(path):
    """Read file and encode to base64."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_b64_safe(path):
    """Get base64 with fallback to parent directory."""
    if os.path.exists(path):
        return get_b64(path)
    elif os.path.exists("../" + path):
        return get_b64("../" + path)
    else:
        raise FileNotFoundError(f"Could not find {path}")


def get_b64_optional(path):
    """Get base64 or return None if file doesn't exist."""
    try:
        return get_b64_safe(path)
    except FileNotFoundError:
        return None


def generate_notebook():
    """Generate the baseline comparison notebook."""
    
    print("  - Reading source files...")
    
    # Read all required files (small files only, skip large SQuAD data)
    main_b64 = get_b64_safe("src/main.py")
    embedders_b64 = get_b64_safe("projects/hybrid_titans/common/embedders.py")
    titan_utils_b64 = get_b64_safe("projects/hybrid_titans/common/titan_utils.py")
    llm_utils_b64 = get_b64_safe("projects/hybrid_titans/common/llm_utils.py")
    text_utils_b64 = get_b64_safe("projects/hybrid_titans/common/text_utils.py")
    eval_utils_b64 = get_b64_safe("projects/hybrid_titans/common/eval_utils.py")
    baselines_b64 = get_b64_safe("projects/hybrid_titans/baselines.py")
    compare_b64 = get_b64_safe("projects/hybrid_titans/compare_baselines.py")
    essays_b64 = get_b64_safe("data/sample_essays.py")
    init_b64 = base64.b64encode(b"# Common modules").decode("utf-8")
    
    rag_infer_b64 = get_b64_optional("is_RAG_able/rag_infer.py")
    init_empty_b64 = base64.b64encode(b"").decode("utf-8")
    
    # Check if processed_squad.py exists (but don't embed it - too large!)
    squad_exists = os.path.exists("data/processed_squad.py") or os.path.exists("../data/processed_squad.py")
    
    print("  - Building notebook cells...")
    
    cells = []
    
    # ===== Cell 1: Title =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Baseline Comparison: PureRAG vs TitanOnly vs HybridRAG vs AdaptiveHybridRAG\n",
            "\n",
            "This notebook compares retrieval strategies:\n",
            "- **PureRAG**: BM25 + Embedding (no Memory)\n",
            "- **TitanOnly**: Memory-guided retrieval only\n", 
            "- **HybridRAG**: BM25 + Memory + Embedding fusion\n",
            "- **AdaptiveHybridRAG**: Per-Chunk Soft Routing (XGBoost scorer)\n",
            "\n",
            "### Modes:\n",
            "1. **Sample Essays Mode**: Use built-in climate/ai/space essays\n",
            "2. **SQuAD Mode**: Use processed SQuAD dataset (single-doc per title)\n",
            "3. **Multi-Doc Mode**: Digest multiple articles into shared memory for cross-document retrieval\n",
            "4. **HotpotQA Mode**: Multi-hop QA across multiple supporting documents (already multi-document)\n",
            "5. **MuSiQue Mode**: Multi-hop QA with 2-4 hop reasoning across 20 paragraphs"
        ]
    })
    
    # ===== Cell 2: Install Dependencies =====
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
            "# Install required packages\n",
            "!pip install -q torch sentence-transformers transformers tqdm scikit-learn xgboost joblib"
        ]
    })
    
    # ===== Cell 3: Setup File System (small files only) =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Setup File System (Core Files)"]
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
        "    with open(target_path, 'wb') as f:\n",
        "        f.write(base64.b64decode(b64_content))\n",
        "    print(f'✓ {path}')\n",
        "\n",
        "# Core files (embedded for convenience)\n",
        "files = {\n",
        f"    'src/main.py': '{main_b64}',\n",
        f"    'data/sample_essays.py': '{essays_b64}',\n",
        f"    'projects/hybrid_titans/common/__init__.py': '{init_b64}',\n",
        f"    'projects/hybrid_titans/common/embedders.py': '{embedders_b64}',\n",
        f"    'projects/hybrid_titans/common/titan_utils.py': '{titan_utils_b64}',\n",
        f"    'projects/hybrid_titans/common/llm_utils.py': '{llm_utils_b64}',\n",
        f"    'projects/hybrid_titans/common/text_utils.py': '{text_utils_b64}',\n",
        f"    'projects/hybrid_titans/common/eval_utils.py': '{eval_utils_b64}',\n",
        f"    'projects/hybrid_titans/baselines.py': '{baselines_b64}',\n",
        f"    'projects/hybrid_titans/compare_baselines.py': '{compare_b64}'\n"
    ]
    
    if rag_infer_b64:
        setup_code.append(f"    ,'is_RAG_able/rag_infer.py': '{rag_infer_b64}'\n")
        setup_code.append(f"    ,'is_RAG_able/__init__.py': '{init_empty_b64}'\n")
        
    setup_code.extend([
        "}\n",
        "\n",
        "print(f'Setting up {len(files)} core files...')\n",
        "for path, content in files.items():\n",
        "    write_file(path, content)\n",
        "\n",
        "print('\\n✅ Core files ready!')\n"
    ])
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_code
    })
    
    # ===== Cell 3.1: Setup XGBoost Scorer Model =====
    xgb_model_b64 = get_b64_optional("is_RAG_able/models/xgb_model.pkl")
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2.1 Setup XGBoost Scorer Model\n",
                   "The XGBoost scorer model is used by AdaptiveHybridRAG for per-chunk soft routing.\n",
                   "It predicts how likely a (question, chunk) pair is retrievable by traditional RAG."]
    })
    
    if xgb_model_b64:
        xgb_setup_source = [
            "import os, base64\n",
            "\n",
            "model_dir = 'is_RAG_able/models'\n",
            "model_path = os.path.join(model_dir, 'xgb_model.pkl')\n",
            "os.makedirs(model_dir, exist_ok=True)\n",
            "\n",
            f"xgb_b64 = '{xgb_model_b64}'\n",
            "with open(model_path, 'wb') as f:\n",
            "    f.write(base64.b64decode(xgb_b64))\n",
            "size_mb = os.path.getsize(model_path) / (1024 * 1024)\n",
            "print(f'✅ XGBoost model written: {model_path} ({size_mb:.1f} MB)')\n"
        ]
    else:
        xgb_setup_source = [
            "import os\n",
            "\n",
            "model_dir = 'is_RAG_able/models'\n",
            "model_path = os.path.join(model_dir, 'xgb_model.pkl')\n",
            "os.makedirs(model_dir, exist_ok=True)\n",
            "\n",
            "if not os.path.exists(model_path):\n",
            "    print(f'⚠️ XGBoost model not found at {model_path}')\n",
            "    print('Please upload xgb_model.pkl to the colab filesystem at that path.')\n",
            "else:\n",
            "    size_mb = os.path.getsize(model_path) / (1024 * 1024)\n",
            "    print(f'✅ Found model: {model_path} ({size_mb:.1f} MB)')\n"
        ]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": xgb_setup_source
    })
    
    # ===== Cell 3.5: Setup SQuAD Data (auto-generate) =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2.5 Setup SQuAD Dataset (Optional)\n",
            "\n",
            "This cell will automatically download and generate the SQuAD dataset.\n",
            "**Skip this cell if you only want to use Sample Essays mode.**"
        ]
    })
    
    # Read process_squad_data.py and embed it
    process_squad_b64 = get_b64_safe("data/process_squad_data.py")
    
    squad_setup_code = [
        "import os\n",
        "import base64\n",
        "\n",
        "# Check if SQuAD data already exists\n",
        "if os.path.exists('data/processed_squad.py'):\n",
        "    size_mb = os.path.getsize('data/processed_squad.py') / (1024 * 1024)\n",
        "    print(f'✅ SQuAD dataset already exists ({size_mb:.1f} MB)')\n",
        "else:\n",
        "    print('📥 Downloading SQuAD dataset...')\n",
        "    !wget -q --show-progress https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O data/train-v2.0.json\n",
        "    \n",
        "    # Write the processing script\n",
        f"    script_b64 = '{process_squad_b64}'\n",
        "    os.makedirs('data', exist_ok=True)\n",
        "    with open('data/process_squad_data.py', 'wb') as f:\n",
        "        f.write(base64.b64decode(script_b64))\n",
        "    \n",
        "    print('⚙️ Processing SQuAD data...')\n",
        "    !python data/process_squad_data.py\n",
        "    \n",
        "    if os.path.exists('data/processed_squad.py'):\n",
        "        size_mb = os.path.getsize('data/processed_squad.py') / (1024 * 1024)\n",
        "        print(f'✅ SQuAD dataset generated ({size_mb:.1f} MB)')\n",
        "    else:\n",
        "        print('❌ Failed to generate SQuAD dataset')\n"
    ]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": squad_setup_code
    })
    
    # ===== Cell 4: Run Sample Essays Comparison =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Run Baseline Comparison (Sample Essays)\n",
            "\n",
            "This cell runs comparison on the Climate essay with 50 epochs."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run baseline comparison on sample essays\n",
            "!python projects/hybrid_titans/compare_baselines.py --essay climate --epochs 50"
        ]
    })
    
    # ===== Cell 5: Run SQuAD Comparison =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Run Baseline Comparison (SQuAD Dataset)\n",
            "\n",
            "This cell runs comparison on the SQuAD dataset.\n",
            "\n",
            "**Parameters:**\n",
            "- `--squad`: Enable SQuAD mode\n",
            "- `--titles N`: Number of titles to evaluate (default: 10)\n",
            "- `--max-questions N`: Max questions per title (default: 5)\n",
            "- `--epochs N`: Digestion epochs (default: 50)\n",
            "- `--verbose`: Show detailed output"
        ]
    })
    
    squad_cell_source = [
        "# Run baseline comparison on SQuAD dataset\n",
        "# Uncomment the line below to run\n",
        "\n"
    ]
    
    if squad_exists:
        squad_cell_source.append("!python projects/hybrid_titans/compare_baselines.py --squad --titles 5 --max-questions 50 --epochs 200 --verbose --save-results\n")
    else:
        squad_cell_source.append("# Note: SQuAD data not set up. Run cell 2.5 first to set up SQuAD.\n")
        squad_cell_source.append("# !python projects/hybrid_titans/compare_baselines.py --squad --titles 5 --max-questions 50 --epochs 200 --verbose --save-results\n")
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": squad_cell_source
    })
    
    # ===== Cell 6: Multi-Document Comparison =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Run Multi-Document Comparison\n",
            "\n",
            "This mode digests multiple SQuAD articles into a **shared** Titan memory, ",
            "enabling cross-document retrieval.\n",
            "\n",
            "**Parameters:**\n",
            "- `--multi-doc`: Enable multi-document mode\n",
            "- `--group-size N`: Number of titles to group together (default: 5)\n",
            "- `--max-questions N`: Max questions per title (default: 5)\n",
            "- `--epochs N`: Digestion epochs (default: 50)"
        ]
    })

    multidoc_cell_source = [
        "# Run multi-document cross-article comparison\n",
        "# Articles are digested into a shared memory, retrieval spans all chunks\n",
        "\n"
    ]

    if squad_exists:
        multidoc_cell_source.append("!python projects/hybrid_titans/compare_baselines.py --multi-doc --group-size 5 --max-questions 5 --epochs 200 --verbose --save-results\n")
    else:
        multidoc_cell_source.append("# Note: SQuAD data not set up. Run cell 2.5 first.\n")
        multidoc_cell_source.append("# !python projects/hybrid_titans/compare_baselines.py --multi-doc --group-size 5 --max-questions 5 --epochs 200 --verbose --save-results\n")

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": multidoc_cell_source
    })

    # ===== Cell: Setup HotpotQA Data =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5.5 Setup HotpotQA Dataset (Optional)\n",
            "\n",
            "This cell will automatically download and generate the HotpotQA dataset.\n",
            "HotpotQA is a **multi-hop QA** dataset requiring reasoning across multiple documents.\n",
            "\n",
            "**Skip this cell if you only want to use Sample Essays or SQuAD mode.**"
        ]
    })

    # Read process_hotpotqa_data.py and embed it
    process_hotpotqa_b64 = get_b64_optional("data/process_hotpotqa_data.py")

    hotpotqa_setup_code = [
        "import os\n",
        "import base64\n",
        "\n",
        "# Check if HotpotQA data already exists\n",
        "if os.path.exists('data/processed_hotpotqa.py'):\n",
        "    size_mb = os.path.getsize('data/processed_hotpotqa.py') / (1024 * 1024)\n",
        "    print(f'\u2705 HotpotQA dataset already exists ({size_mb:.1f} MB)')\n",
        "else:\n",
        "    print('\ud83d\udce5 Downloading HotpotQA dataset...')\n",
        "    !wget -q --show-progress http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json -O data/hotpot_train_v1.1.json\n",
        "    \n",
    ]

    if process_hotpotqa_b64:
        hotpotqa_setup_code.extend([
            "    # Write the processing script\n",
            f"    script_b64 = '{process_hotpotqa_b64}'\n",
            "    os.makedirs('data', exist_ok=True)\n",
            "    with open('data/process_hotpotqa_data.py', 'wb') as f:\n",
            "        f.write(base64.b64decode(script_b64))\n",
            "    \n",
        ])
    else:
        hotpotqa_setup_code.append("    # Note: process_hotpotqa_data.py not found locally, ensure it exists in data/\n")

    hotpotqa_setup_code.extend([
        "    print('\u2699\ufe0f Processing HotpotQA data...')\n",
        "    !python data/process_hotpotqa_data.py\n",
        "    \n",
        "    if os.path.exists('data/processed_hotpotqa.py'):\n",
        "        size_mb = os.path.getsize('data/processed_hotpotqa.py') / (1024 * 1024)\n",
        "        print(f'\u2705 HotpotQA dataset generated ({size_mb:.1f} MB)')\n",
        "    else:\n",
        "        print('\u274c Failed to generate HotpotQA dataset')\n"
    ])

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": hotpotqa_setup_code
    })

    # ===== Cell: Run HotpotQA Comparison =====
    hotpotqa_exists = os.path.exists("data/processed_hotpotqa.py") or os.path.exists("../data/processed_hotpotqa.py")

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Run Baseline Comparison (HotpotQA)\n",
            "\n",
            "This cell runs comparison on the HotpotQA dataset (multi-hop QA).\n",
            "\n",
            "**Parameters:**\n",
            "- `--hotpotqa`: Enable HotpotQA mode\n",
            "- `--titles N`: Number of titles to evaluate (default: 10)\n",
            "- `--max-questions N`: Max questions per title (default: 5)\n",
            "- `--epochs N`: Digestion epochs (default: 50)\n",
            "- `--verbose`: Show detailed output"
        ]
    })
    

    hotpotqa_cell_source = [
        "# Run baseline comparison on HotpotQA dataset (multi-hop QA)\n",
        "\n"
    ]

    if hotpotqa_exists:
        hotpotqa_cell_source.append("!python projects/hybrid_titans/compare_baselines.py --hotpotqa --titles 50 --max-questions 5 --epochs 300 --verbose --save-results\n")
    else:
        hotpotqa_cell_source.append("# Note: HotpotQA data not set up. Run cell 5.5 first to set up HotpotQA.\n")
        hotpotqa_cell_source.append("# !python projects/hybrid_titans/compare_baselines.py --hotpotqa --titles 50 --max-questions 5 --epochs 300 --verbose --save-results\n")

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": hotpotqa_cell_source
    })

    # ===== Cell: Setup MuSiQue Data =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6.5 Setup MuSiQue Dataset (Optional)\n",
            "\n",
            "This cell will automatically download and generate the MuSiQue dataset.\n",
            "MuSiQue is a **multi-hop QA** dataset requiring reasoning across 2-4 hops, ",
            "with 20 paragraphs per question.\n",
            "\n",
            "**Skip this cell if you only want to use other modes.**"
        ]
    })

    # Read process_MuSiQue_data.py and embed it
    process_musique_b64 = get_b64_optional("data/process_MuSiQue_data.py")

    musique_setup_code = [
        "import os\n",
        "import base64\n",
        "\n",
        "# Check if MuSiQue data already exists\n",
        "if os.path.exists('data/processed_musique.py'):\n",
        "    size_mb = os.path.getsize('data/processed_musique.py') / (1024 * 1024)\n",
        "    print(f'\u2705 MuSiQue dataset already exists ({size_mb:.1f} MB)')\n",
        "else:\n",
        "    print('\ud83d\udce5 Setting up MuSiQue dataset...')\n",
    ]

    if process_musique_b64:
        musique_setup_code.extend([
            "    # Write the processing script\n",
            f"    script_b64 = '{process_musique_b64}'\n",
            "    os.makedirs('data', exist_ok=True)\n",
            "    with open('data/process_MuSiQue_data.py', 'wb') as f:\n",
            "        f.write(base64.b64decode(script_b64))\n",
            "    \n",
        ])
    else:
        musique_setup_code.append("    # Note: process_MuSiQue_data.py not found locally, ensure it exists in data/\n")

    musique_setup_code.extend([
        "    print('\u2699\ufe0f Processing MuSiQue data...')\n",
        "    !python data/process_MuSiQue_data.py\n",
        "    \n",
        "    if os.path.exists('data/processed_musique.py'):\n",
        "        size_mb = os.path.getsize('data/processed_musique.py') / (1024 * 1024)\n",
        "        print(f'\u2705 MuSiQue dataset generated ({size_mb:.1f} MB)')\n",
        "    else:\n",
        "        print('\u274c Failed to generate MuSiQue dataset')\n"
    ])

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": musique_setup_code
    })

    # ===== Cell: Run MuSiQue Comparison =====
    musique_exists = os.path.exists("data/processed_musique.py") or os.path.exists("../data/processed_musique.py")

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Run Baseline Comparison (MuSiQue)\n",
            "\n",
            "This cell runs comparison on the MuSiQue dataset (multi-hop QA, 2-4 hops).\n",
            "\n",
            "**Parameters:**\n",
            "- `--musique`: Enable MuSiQue mode\n",
            "- `--titles N`: Number of titles to evaluate (default: 10)\n",
            "- `--max-questions N`: Max questions per title (default: 5)\n",
            "- `--epochs N`: Digestion epochs (default: 50)\n",
            "- `--verbose`: Show detailed output"
        ]
    })

    musique_cell_source = [
        "# Run baseline comparison on MuSiQue dataset (multi-hop QA)\n",
        "\n"
    ]

    if musique_exists:
        musique_cell_source.append("!python projects/hybrid_titans/compare_baselines.py --musique --titles 50 --max-questions 5 --epochs 300 --verbose --save-results\n")
    else:
        musique_cell_source.append("# Note: MuSiQue data not set up. Run cell 6.5 first to set up MuSiQue.\n")
        musique_cell_source.append("# !python projects/hybrid_titans/compare_baselines.py --musique --titles 50 --max-questions 5 --epochs 300 --verbose --save-results\n")

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": musique_cell_source
    })

    # ===== Cell: Custom Comparison (Optional) =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Custom Comparison (Optional)\n",
            "\n",
            "Run the comparison with your own parameters."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Custom parameters for sample essays\n",
            "# !python projects/hybrid_titans/compare_baselines.py --essay ai --epochs 100 --topk 3\n",
            "\n",
            "# Custom parameters for SQuAD (single-doc)\n",
            "# !python projects/hybrid_titans/compare_baselines.py --squad --titles 20 --max-questions 10 --epochs 50 --verbose\n",
            "\n",
            "# Custom parameters for multi-doc\n",
            "# !python projects/hybrid_titans/compare_baselines.py --multi-doc --group-size 10 --max-questions 5 --epochs 50 --verbose\n",
            "\n",
            "# Custom parameters for HotpotQA\n",
            "# !python projects/hybrid_titans/compare_baselines.py --hotpotqa --titles 20 --max-questions 10 --epochs 300 --verbose\n",
            "\n",
            "# Custom parameters for MuSiQue\n",
            "# !python projects/hybrid_titans/compare_baselines.py --musique --titles 20 --max-questions 10 --epochs 300 --verbose"
        ]
    })
    
    # --- Cell: Download Evaluation Results ---
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Download Evaluation Results\n",
            "\n",
            "Download saved JSON results from the `evaluations/` directory to your local machine."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os, glob, shutil\n",
            "\n",
            "eval_files = sorted(glob.glob('evaluations/*.json'))\n",
            "if not eval_files:\n",
            "    print('No evaluation results found in evaluations/')\n",
            "else:\n",
            "    print(f'Found {len(eval_files)} result file(s):')\n",
            "    for f in eval_files:\n",
            "        print(f'  - {f}')\n",
            "    try:\n",
            "        from google.colab import files\n",
            "        shutil.make_archive('evaluation_results', 'zip', '.', 'evaluations')\n",
            "        files.download('evaluation_results.zip')\n",
            "        print('Downloading evaluation_results.zip ...')\n",
            "    except ImportError:\n",
            "        print('Not on Colab. Files at:', os.path.abspath('evaluations/'))"
        ]
    })


    # Notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def main():
    """Main entry point."""
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    print("Generating Baseline Comparison Notebook...")
    notebook = generate_notebook()
    
    # Write notebook
    output_path = os.path.join(project_root, "notebooks", "Baseline_Comparison.ipynb")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Generated: {output_path}")


if __name__ == "__main__":
    main()
