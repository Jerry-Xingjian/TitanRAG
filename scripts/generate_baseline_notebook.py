#!/usr/bin/env python3
"""
Generate Baseline Comparison Notebook.

This script generates a Jupyter notebook that compares the three
baseline retrieval strategies: PureRAG, TitanOnly, and HybridRAG.

Structure similar to Titan_Experiment.ipynb:
1. Install Dependencies
2. Setup File System (creates all files in one cell)
3. Run Baseline Comparison (one cell!)

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


def generate_notebook():
    """Generate the baseline comparison notebook."""
    
    # Read all required files
    main_b64 = get_b64_safe("src/main.py")
    embedders_b64 = get_b64_safe("projects/hybrid_titans/common/embedders.py")
    titan_utils_b64 = get_b64_safe("projects/hybrid_titans/common/titan_utils.py")
    llm_utils_b64 = get_b64_safe("projects/hybrid_titans/common/llm_utils.py")
    text_utils_b64 = get_b64_safe("projects/hybrid_titans/common/text_utils.py")
    baselines_b64 = get_b64_safe("projects/hybrid_titans/baselines.py")
    compare_b64 = get_b64_safe("projects/hybrid_titans/compare_baselines.py")
    essays_b64 = get_b64_safe("data/sample_essays.py")
    init_b64 = base64.b64encode(b"# Common modules").decode("utf-8")
    
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
            "- **HybridRAG**: BM25 + Memory + Embedding fusion"
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
    
    # ===== Cell 3: Setup File System =====
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
        "    with open(target_path, 'wb') as f:\n",
        "        f.write(base64.b64decode(b64_content))\n",
        "    print(f'Created {target_path}')\n",
        "\n",
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
        "for path, content in files.items():\n",
        "    write_file(path, content)\n",
        "\n",
        "print('\\n✅ All files created!')\n"
    ]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_code
    })
    
    # ===== Cell 4: Run Comparison =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Run Baseline Comparison\n",
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
            "# Run baseline comparison\n",
            "!python projects/hybrid_titans/compare_baselines.py --essay climate --epochs 50"
        ]
    })
    
    # ===== Cell 5: Custom Comparison (Optional) =====
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Custom Comparison (Optional)\n",
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
            "# Custom parameters\n",
            "# !python projects/hybrid_titans/compare_baselines.py --essay climate --epochs 100 --topk 5"
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
