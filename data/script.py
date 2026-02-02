"""
Script to clean and process train.csv dataset.

This script:
1. Filters out unique titles
2. Combines duplicate contexts based on title
3. Extracts question and answer pairs for each context

Output format matches sample_essays.py structure.
"""

import csv
import ast
import re
import sys
import io

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
        data: List of row dictionaries from CSV
        
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
            # Escape triple quotes in content
            escaped_context = context.replace('"""', '\\"\\"\\"')
            f.write(f'{var_name} = """\n{escaped_context}\n"""\n\n')
        
        # Generate questions dictionary
        f.write('# ============================================================\n')
        f.write('# TEST_QUESTIONS - Question-Answer pairs by title\n')
        f.write('# ============================================================\n\n')
        f.write('TEST_QUESTIONS = {\n')
        
        for title, qa_pairs in questions.items():
            if not qa_pairs:
                continue
            # Escape quotes in title
            escaped_title = title.replace('"', '\\"')
            f.write(f'    "{escaped_title}": [\n')
            
            for question, answer in qa_pairs:
                # Escape quotes in question and answer
                escaped_q = question.replace('"', '\\"').replace('\n', ' ')
                escaped_a = answer.replace('"', '\\"').replace('\n', ' ')
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
            escaped_title = title.replace('"', '\\"')
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
    # Configuration
    INPUT_CSV = "train.csv"
    OUTPUT_FILE = "processed_squad.py"
    
    print("=" * 60)
    print("SQuAD Dataset Processor")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading CSV data...")
    data = load_csv_data(INPUT_CSV)
    
    # Process data
    print("\n[2/3] Processing dataset...")
    contexts, questions = process_dataset(data)
    
    # Generate output
    print("\n[3/3] Generating output file...")
    generate_output_file(contexts, questions, OUTPUT_FILE)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  - Unique titles: {len(contexts)}")
    print(f"  - Total QA pairs: {sum(len(qa) for qa in questions.values())}")
    print(f"  - Output file: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
