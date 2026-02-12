"""
Script to clean and process HotpotQA dataset (JSON format).

This script:
1. Extracts supporting documents from multi-document contexts
2. Combines supporting documents into a unified context
3. Creates unique titles based on supporting facts
4. Extracts question-answer pairs

Output format matches sample_essays.py and processed_squad.py structure.

Key differences from Squad 2.0:
- Multi-document contexts (need to combine)
- Supporting facts annotation (use to filter distractors)
- Yes/No answers in addition to text extraction
- Multi-hop reasoning questions

Download HotpotQA dataset:
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
"""

import json
import re
import os
import sys
from collections import defaultdict


def load_hotpotqa_json(file_path: str) -> list:
    """
    Load HotpotQA JSON dataset.
    
    Args:
        file_path: Path to the JSON file (hotpot_train_v1.1.json or hotpot_dev_distractor_v1.json)
        
    Returns:
        List of data items, each containing:
        - _id: unique identifier
        - question: the question text
        - answer: answer (text or yes/no)
        - supporting_facts: [[title, sent_id], ...]
        - context: [[title, [sent1, sent2, ...]], ...]
        - type: question type (comparison, bridge, etc.)
        - level: difficulty (easy, medium, hard)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} items from HotpotQA dataset")
    return data


def extract_supporting_context(item: dict, include_all: bool = False) -> str:
    """
    Extract and combine context from supporting documents.
    
    Args:
        item: HotpotQA data item
        include_all: If True, include all documents (including distractors)
                    If False, only include supporting documents
        
    Returns:
        Combined context string with document titles as headers
    """
    if include_all:
        # Include all documents
        context_parts = []
        for doc_title, sentences in item['context']:
            paragraph = " ".join(sentences)
            context_parts.append(f"# {doc_title}\n{paragraph}")
    else:
        # Only include supporting documents
        supporting_titles = set([title for title, _ in item['supporting_facts']])
        context_parts = []
        
        for doc_title, sentences in item['context']:
            if doc_title in supporting_titles:
                paragraph = " ".join(sentences)
                context_parts.append(f"# {doc_title}\n{paragraph}")
    
    return "\n\n".join(context_parts)


def create_unique_title(item: dict) -> str:
    """
    Create a unique title for each question based on supporting facts.
    
    Args:
        item: HotpotQA data item
        
    Returns:
        Unique title string combining supporting document titles
    """
    # Extract supporting document titles
    supporting_titles = []
    seen = set()
    for title, _ in item['supporting_facts']:
        if title not in seen:
            supporting_titles.append(title)
            seen.add(title)
    
    # Combine titles (max 3 to keep variable names reasonable)
    if len(supporting_titles) <= 3:
        return " and ".join(supporting_titles)
    else:
        return " and ".join(supporting_titles[:3]) + " et al"


def process_hotpotqa_dataset(data: list, 
                             include_all_docs: bool = False,
                             max_items: int = None) -> tuple:
    """
    Process HotpotQA dataset into contexts and questions.
    
    Args:
        data: List of HotpotQA items
        include_all_docs: If True, include distractor documents
        max_items: Maximum number of items to process (for testing)
        
    Returns:
        Tuple of (contexts_dict, questions_dict, metadata_dict)
        - contexts_dict: {title: combined_context}
        - questions_dict: {title: [(question, answer), ...]}
        - metadata_dict: {title: {'type': [...], 'level': [...], 'ids': [...]}}
    """
    # Use defaultdict to collect items with same supporting documents
    title_to_items = defaultdict(list)
    
    # Limit items if specified
    if max_items:
        data = data[:max_items]
    
    # Group items by unique title
    for item in data:
        title = create_unique_title(item)
        title_to_items[title].append(item)
    
    # Process each group
    contexts_dict = {}
    questions_dict = {}
    metadata_dict = {}
    
    for title, items in title_to_items.items():
        # Use the first item's context (they should be similar if supporting docs are the same)
        context = extract_supporting_context(items[0], include_all=include_all_docs)
        contexts_dict[title] = context
        
        # Collect all Q&A pairs
        qa_pairs = []
        types = []
        levels = []
        ids = []
        
        for item in items:
            question = item['question']
            answer = item['answer']
            qa_pairs.append((question, answer))
            
            # Collect metadata
            types.append(item.get('type', 'unknown'))
            levels.append(item.get('level', 'unknown'))
            ids.append(item['_id'])
        
        questions_dict[title] = qa_pairs
        metadata_dict[title] = {
            'type': types,
            'level': levels,
            'ids': ids
        }
    
    print(f"Processed {len(contexts_dict)} unique context groups")
    print(f"Total QA pairs: {sum(len(qa) for qa in questions_dict.values())}")
    
    return contexts_dict, questions_dict, metadata_dict


def create_variable_name(title: str) -> str:
    """
    Create a valid Python variable name from title.
    
    Args:
        title: Original title string
        
    Returns:
        Valid Python variable name
    """
    # Replace non-alphanumeric characters with underscore
    name = re.sub(r'[^a-zA-Z0-9]', '_', title)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Ensure starts with letter and add prefix
    if name and name[0].isdigit():
        name = 'CONTEXT_' + name
    else:
        name = 'CONTEXT_' + name.upper()
    return name


def generate_output_file(contexts: dict, 
                         questions: dict, 
                         metadata: dict,
                         output_path: str,
                         include_metadata: bool = True):
    """
    Generate output Python file in sample_essays.py format.
    
    Args:
        contexts: Dictionary of {title: combined_context}
        questions: Dictionary of {title: [(question, answer), ...]}
        metadata: Dictionary of {title: {'type': [...], 'level': [...], 'ids': [...]}}
        output_path: Path for output file
        include_metadata: Whether to include metadata comments
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write('"""\n')
        f.write('Processed HotpotQA Dataset for TitanRAG Experiments\n')
        f.write('\n')
        f.write('This file contains processed contexts and question-answer pairs\n')
        f.write('extracted from the HotpotQA dataset.\n')
        f.write('\n')
        f.write('Note: HotpotQA is a multi-hop QA dataset requiring reasoning\n')
        f.write('across multiple documents. Contexts combine supporting documents.\n')
        f.write('"""\n\n')
        
        # Generate context variables
        f.write('# ============================================================\n')
        f.write('# CONTEXTS - Combined supporting documents\n')
        f.write('# ============================================================\n\n')
        
        for title, context in contexts.items():
            var_name = create_variable_name(title)
            
            # Add metadata comment if requested
            if include_metadata and title in metadata:
                meta = metadata[title]
                f.write(f'# Question types: {", ".join(set(meta["type"]))}\n')
                f.write(f'# Difficulty levels: {", ".join(set(meta["level"]))}\n')
                f.write(f'# Number of questions: {len(questions[title])}\n')
            
            # Escape backslashes first, then triple quotes in content
            escaped_context = context.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
            f.write(f'{var_name} = """\n{escaped_context}\n"""\n\n')
        
        # Generate questions dictionary
        f.write('# ============================================================\n')
        f.write('# TEST_QUESTIONS - Question-Answer pairs\n')
        f.write('# ============================================================\n\n')
        f.write('TEST_QUESTIONS = {\n')
        
        for title, qa_pairs in questions.items():
            if not qa_pairs:
                continue
            
            # Escape quotes in title (backslashes first, then quotes)
            escaped_title = title.replace('\\', '\\\\').replace('"', '\\"')
            f.write(f'    "{escaped_title}": [\n')
            
            for question, answer in qa_pairs:
                # Escape backslashes first, then quotes and newlines
                escaped_q = question.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
                escaped_a = answer.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
                f.write(f'        ("{escaped_q}", "{escaped_a}"),\n')
            
            f.write('    ],\n')
        
        f.write('}\n\n')
        
        # Optional: Include metadata dictionary
        if include_metadata:
            f.write('# ============================================================\n')
            f.write('# METADATA - Question types and difficulty levels\n')
            f.write('# ============================================================\n\n')
            f.write('QUESTION_METADATA = {\n')
            
            for title, meta in metadata.items():
                escaped_title = title.replace('\\', '\\\\').replace('"', '\\"')
                f.write(f'    "{escaped_title}": {{\n')
                f.write(f'        "types": {list(set(meta["type"]))},\n')
                f.write(f'        "levels": {list(set(meta["level"]))},\n')
                f.write(f'        "ids": {meta["ids"]},\n')
                f.write('    },\n')
            
            f.write('}\n\n')
        
        # Helper functions
        f.write('# ============================================================\n')
        f.write('# HELPER FUNCTIONS\n')
        f.write('# ============================================================\n\n')
        
        f.write('def get_all_contexts():\n')
        f.write('    """Return all contexts as a dictionary."""\n')
        f.write('    return {\n')
        for title in contexts.keys():
            var_name = create_variable_name(title)
            escaped_title = title.replace('\\', '\\\\').replace('"', '\\"')
            f.write(f'        "{escaped_title}": {var_name},\n')
        f.write('    }\n\n')
        
        f.write('def get_test_questions():\n')
        f.write('    """Return test questions with expected answers."""\n')
        f.write('    return TEST_QUESTIONS\n\n')
        
        if include_metadata:
            f.write('def get_question_metadata():\n')
            f.write('    """Return metadata for questions (types, levels, IDs)."""\n')
            f.write('    return QUESTION_METADATA\n')
    
    print(f"Output saved to: {output_path}")


def main():
    """Main function to process HotpotQA dataset."""
    # Configuration
    INPUT_FILE = "hotpot_train_v1.1.json"  # or hotpot_dev_distractor_v1.json
    OUTPUT_FILE = "processed_hotpotqa.py"
    
    # Processing options
    INCLUDE_ALL_DOCS = False  # False = only supporting docs, True = include distractors
    INCLUDE_METADATA = True   # Include metadata comments and dictionary
    MAX_ITEMS = None          # Set to e.g., 1000 for testing, None for all
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    
    print("=" * 60)
    print("HotpotQA Dataset Processor")
    print("=" * 60)
    print(f"Mode: {'All documents (incl. distractors)' if INCLUDE_ALL_DOCS else 'Supporting documents only'}")
    print(f"Metadata: {'Included' if INCLUDE_METADATA else 'Excluded'}")
    if MAX_ITEMS:
        print(f"Processing: First {MAX_ITEMS} items")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"\n❌ Error: Input file not found!")
        print(f"   Expected: {input_path}")
        print(f"\n   To download HotpotQA dataset:")
        print(f"   wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json")
        print(f"   wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json")
        return
    
    # Load data
    print("\n[1/3] Loading data...")
    data = load_hotpotqa_json(input_path)
    
    # Process data
    print("\n[2/3] Processing dataset...")
    contexts, questions, metadata = process_hotpotqa_dataset(
        data, 
        include_all_docs=INCLUDE_ALL_DOCS,
        max_items=MAX_ITEMS
    )
    
    # Generate output
    print("\n[3/3] Generating output file...")
    generate_output_file(
        contexts, 
        questions, 
        metadata,
        output_path,
        include_metadata=INCLUDE_METADATA
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  - Unique context groups: {len(contexts)}")
    print(f"  - Total QA pairs: {sum(len(qa) for qa in questions.values())}")
    print(f"  - Output file: {output_path}")
    
    # Show question type distribution
    if metadata:
        all_types = []
        all_levels = []
        for meta in metadata.values():
            all_types.extend(meta['type'])
            all_levels.extend(meta['level'])
        
        from collections import Counter
        type_counts = Counter(all_types)
        level_counts = Counter(all_levels)
        
        print(f"\n  Question Types:")
        for qtype, count in type_counts.items():
            print(f"    - {qtype}: {count}")
        
        print(f"\n  Difficulty Levels:")
        for level, count in level_counts.items():
            print(f"    - {level}: {count}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
