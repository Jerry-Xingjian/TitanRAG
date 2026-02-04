#!/usr/bin/env python3
"""
Generate Baseline Comparison Notebook.

This script generates a Jupyter notebook that compares the three
baseline retrieval strategies: PureRAG, TitanOnly, and HybridRAG.

Structure similar to Titan_Experiment.ipynb:
1. Install Dependencies
2. Setup File System (creates all files in one cell)
3. Run Baseline Comparison (sample essays or SQuAD)

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
    baselines_b64 = get_b64_safe("projects/hybrid_titans/baselines.py")
    compare_b64 = get_b64_safe("projects/hybrid_titans/compare_baselines.py")
    essays_b64 = get_b64_safe("data/sample_essays.py")
    init_b64 = base64.b64encode(b"# Common modules").decode("utf-8")
    
    # Check if processed_squad.py exists (but don't embed it - too large!)
    squad_exists = os.path.exists("data/processed_squad.py") or os.path.exists("../data/processed_squad.py")
    
    print("  - Building notebook cells...")
    
    cells = []
    
    # ===== Cell 1: Title =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Baseline Comparison: PureRAG vs TitanOnly vs HybridRAG\n",
            "\n",
            "This notebook compares three retrieval strategies:\n",
            "- **PureRAG**: BM25 + Embedding (no Memory)\n",
            "- **TitanOnly**: Memory-guided retrieval only\n", 
            "- **HybridRAG**: BM25 + Memory + Embedding fusion\n",
            "\n",
            "### Modes:\n",
            "1. **Sample Essays Mode**: Use built-in climate/ai/space essays\n",
            "2. **SQuAD Mode**: Use processed SQuAD dataset (requires setup)"
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
            "!pip install -q torch sentence-transformers transformers tqdm"
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
        f"    'projects/hybrid_titans/baselines.py': '{baselines_b64}',\n",
        f"    'projects/hybrid_titans/compare_baselines.py': '{compare_b64}'\n",
        "}\n",
        "\n",
        "print(f'Setting up {len(files)} core files...')\n",
        "for path, content in files.items():\n",
        "    write_file(path, content)\n",
        "\n",
        "print('\\n✅ Core files ready!')\n"
    ]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_code
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
        squad_cell_source.append("!python projects/hybrid_titans/compare_baselines.py --squad --titles 5 --max-questions 3 --epochs 20\n")
    else:
        squad_cell_source.append("# Note: SQuAD data not set up. Run cell 2.5 first to set up SQuAD.\n")
        squad_cell_source.append("# !python projects/hybrid_titans/compare_baselines.py --squad --titles 5 --max-questions 3 --epochs 20\n")
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": squad_cell_source
    })
    
    # ===== Cell 6: Custom Comparison (Optional) =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Custom Comparison (Optional)\n",
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
            "# !python projects/hybrid_titans/compare_baselines.py --essay ai --epochs 100 --topk 5\n",
            "\n",
            "# Custom parameters for SQuAD\n",
            "# !python projects/hybrid_titans/compare_baselines.py --squad --titles 20 --max-questions 10 --epochs 50 --verbose"
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
