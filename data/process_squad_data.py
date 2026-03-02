"""
Script to clean and process SQuAD dataset (JSON or CSV format).

This script:
1. Filters out unique titles
2. Combines duplicate contexts based on title
3. Extracts question and answer pairs for each context

Output format matches sample_essays.py structure.

Supports:
- SQuAD v2.0 JSON format (train-v2.0.json)
- CSV format (train.csv)
"""

import csv
import ast
import re
import sys
import io
import json
import os

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_csv_data(file_path: str) -> list:
    """
    Load CSV dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of row dictionaries
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    print(f"Loaded {len(data)} rows from CSV")
    return data


def load_json_data(file_path: str) -> list:
    """
    Load SQuAD v2.0 JSON dataset and convert to row format.
    
    Args:
        file_path: Path to the JSON file (train-v2.0.json)
        
    Returns:
        List of row dictionaries with keys: title, context, question, answers
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    
    data = []
    for article in squad_data['data']:
        title = article['title']
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                # Handle both v1 and v2 format
                answers = qa.get('answers', [])
                if not answers and 'plausible_answers' in qa:
                    answers = qa['plausible_answers']
                
                # Convert to format expected by process_dataset
                answer_text = answers[0]['text'] if answers else ''
                data.append({
                    'title': title,
                    'context': context,
                    'question': question,
                    'answers': answer_text,  # Already parsed
                    '_parsed': True  # Flag to skip parse_answer
                })
    
    print(f"Loaded {len(data)} QA pairs from JSON")
    return data


def parse_answer(answers_str: str) -> str:
    """
    Parse the answer string from CSV format.
    
    Format: {'text': array(['answer text'], dtype=object), 'answer_start': array([123], dtype=int32)}
    
    Args:
        answers_str: Raw answer string from CSV
        
    Returns:
        Extracted answer text, or empty string if parsing fails
    """
    try:
        # Clean numpy array format to Python list format
        cleaned = answers_str.replace("array(", "").replace(", dtype=object)", "").replace(", dtype=int32)", "")
        answers_dict = ast.literal_eval(cleaned)
        
        text_list = answers_dict.get('text', [])
        if text_list and len(text_list) > 0:
            return text_list[0].strip()
    except Exception:
        pass
    return ""


def process_dataset(data: list) -> tuple:
    """
    Process the dataset to extract unique titles, combined contexts, and QA pairs.
    
    Args:
        data: List of row dictionaries from CSV or JSON
        
    Returns:
        Tuple of (contexts_dict, questions_dict)
        - contexts_dict: {title: combined_context}
        - questions_dict: {title: [(question, answer), ...]}
    """
    # Store unique contexts per title (preserve order, avoid duplicates)
    title_contexts = {}  # {title: [list of unique context paragraphs]}
    title_questions = {}  # {title: [(question, answer), ...]}
    
    for row in data:
        title = row['title'].strip()
        context = row['context'].strip()
        question = row['question'].strip()
        
        # Check if answer is pre-parsed (from JSON) or needs parsing (from CSV)
        if row.get('_parsed'):
            answer = row['answers'].strip() if row['answers'] else ''
        else:
            answer = parse_answer(row['answers'])
        
        # Initialize title entry if not exists
        if title not in title_contexts:
            title_contexts[title] = []
            title_questions[title] = []
        
        # Add context if not already present (avoid duplicates)
        if context not in title_contexts[title]:
            title_contexts[title].append(context)
        
        # Add question-answer pair (only if answer exists)
        if answer:
            title_questions[title].append((question, answer))
    
    # Combine contexts for each title
    contexts_dict = {}
    for title, paragraphs in title_contexts.items():
        contexts_dict[title] = "\n\n".join(paragraphs)
    
    print(f"Processed {len(contexts_dict)} unique titles")
    print(f"Total QA pairs: {sum(len(qa) for qa in title_questions.values())}")
    
    return contexts_dict, title_questions


def generate_output_file(contexts: dict, questions: dict, output_path: str):
    """
    Generate output Python file in sample_essays.py format.
    
    Args:
        contexts: Dictionary of {title: combined_context}
        questions: Dictionary of {title: [(question, answer), ...]}
        output_path: Path for output file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write('"""\n')
        f.write('Processed SQuAD Dataset for TitanRAG Experiments\n')
        f.write('\n')
        f.write('This file contains processed contexts and question-answer pairs\n')
        f.write('extracted from the train.csv dataset.\n')
        f.write('"""\n\n')
        
        # Generate context variables
        f.write('# ============================================================\n')
        f.write('# CONTEXTS - Combined paragraphs by title\n')
        f.write('# ============================================================\n\n')
        
        for title, context in contexts.items():
            # Create valid Python variable name
            var_name = create_variable_name(title)
            # Escape backslashes first, then triple quotes in content
            escaped_context = context.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
            f.write(f'{var_name} = """\n{escaped_context}\n"""\n\n')
        
        # Generate questions dictionary
        f.write('# ============================================================\n')
        f.write('# TEST_QUESTIONS - Question-Answer pairs by title\n')
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
        f.write('    return TEST_QUESTIONS\n')
    
    print(f"Output saved to: {output_path}")


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
    # Ensure starts with letter
    if name and name[0].isdigit():
        name = 'CONTEXT_' + name
    else:
        name = 'CONTEXT_' + name.upper()
    return name


def main():
    """Main function to process the dataset."""
    # Configuration - auto-detect input format
    INPUT_JSON = "train-v2.0.json"
    INPUT_CSV = "train.csv"
    OUTPUT_FILE = "processed_squad.py"
    
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    
    print("=" * 60)
    print("SQuAD Dataset Processor")
    print("=" * 60)
    
    # Try to find input file (JSON or CSV)
    json_path = os.path.join(script_dir, INPUT_JSON)
    csv_path = os.path.join(script_dir, INPUT_CSV)
    
    # Load data - prefer JSON (official SQuAD format)
    print("\n[1/3] Loading data...")
    if os.path.exists(json_path):
        print(f"Found JSON file: {INPUT_JSON}")
        data = load_json_data(json_path)
    elif os.path.exists(csv_path):
        print(f"Found CSV file: {INPUT_CSV}")
        data = load_csv_data(csv_path)
    else:
        print(f"❌ Error: No input file found!")
        print(f"   Expected: {INPUT_JSON} or {INPUT_CSV} in {script_dir}")
        print(f"\n   To download SQuAD v2.0:")
        print(f"   wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")
        return
    
    # Process data
    print("\n[2/3] Processing dataset...")
    contexts, questions = process_dataset(data)
    
    # Generate output
    print("\n[3/3] Generating output file...")
    generate_output_file(contexts, questions, output_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  - Unique titles: {len(contexts)}")
    print(f"  - Total QA pairs: {sum(len(qa) for qa in questions.values())}")
    print(f"  - Output file: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
