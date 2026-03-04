"""
Script to strictly clean, and process local MuSiQue dataset.
Specifically tailored for MuSiQue JSONL format (20-paragraph contexts),
converting it to the HotpotQA-style format required by TitanRAG.

Features added:
- Auto-download functionality if the dataset is missing.
- Aligned output generation with process_hotpotqa_data.py.
"""

import json
import re
import os
import sys
import io
import traceback
import urllib.request
from collections import defaultdict
from typing import List, Dict

# Fix Windows console encoding issue
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def download_musique_dataset(output_path: str):
    """Automatically download the MuSiQue dataset if it doesn't exist."""
    # Using the dev set as it's typically the best size for baseline testing
    url = "https://huggingface.co/datasets/StonyBrookNLP/musique/resolve/main/data/musique_ans_v1.0_dev.jsonl"
    print(f"📥 Downloading MuSiQue dataset from {url}...")
    try:
        def progress_bar(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r  - Downloading: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, output_path, reporthook=progress_bar)
        print("\n✅ Download complete!")
    except Exception as e:
        print(f"\n❌ Download failed: {str(e)}")
        print(f"   Please manually download from HuggingFace and save as {output_path}")
        sys.exit(1)


class MuSiQueCleaner:
    def __init__(self):
        self.cleaned_data = []
        self.seen_exact = set()
        self.logs = []
        self.blocklist = ["bad_word1", "bad_word2"]

    def log_action(self, sample_id: str, action: str, reason: str):
        self.logs.append({"sample_id": sample_id, "action": action, "reason": reason})

    def normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text

    def check_compliance(self, text: str) -> bool:
        text_lower = text.lower()
        for bad_word in self.blocklist:
            if bad_word.lower() in text_lower:
                return False
        return True

    def process_dataset(self, data: List[Dict], max_items: int = None) -> List[Dict]:
        self.logs = []

        if max_items:
            data = data[:max_items]

        for idx, item in enumerate(data):
            sample_id = item.get('id', f"musique_{idx}")
            try:
                q = item.get('question', '')
                a = item.get('answer', '')

                # MuSiQue specifically uses 'paragraphs' list
                paragraphs = item.get('paragraphs', [])

                if not q or not a:
                    self.log_action(sample_id, "deleted", "question/answer is empty")
                    continue
                if len(q) < 3:
                    self.log_action(sample_id, "deleted", "question is too short")
                    continue

                item['question'] = self.normalize_text(q)
                item['answer'] = self.normalize_text(str(a))

                if not self.check_compliance(item['question'] + str(item['answer'])):
                    self.log_action(sample_id, "deleted", "triggered blocklist")
                    continue

                # Parse MuSiQue paragraphs to extract supporting facts and context
                valid_sf = []
                sf_titles = set()
                context_docs = []

                for p in paragraphs:
                    title = p.get('title', 'Unknown Title')
                    text = p.get('paragraph_text', '')

                    # Store all context documents (typically 20 in MuSiQue)
                    context_docs.append({'title': title, 'text': text})

                    # Track supporting facts
                    if p.get('is_supporting', False):
                        valid_sf.append({'title': title})
                        sf_titles.add(title)

                # Fallback if no supporting facts are marked
                if not valid_sf and context_docs:
                    fallback_title = context_docs[0]['title']
                    valid_sf.append({'title': fallback_title})
                    sf_titles.add(fallback_title)

                item['supporting_facts'] = valid_sf
                item['context'] = context_docs

                # Calculate hop count (MuSiQue usually has 2-4 hops, default to 2 if unknown)
                item['num_hops'] = item.get('question_decomposition', [])
                hop_count = len(item['num_hops']) if item['num_hops'] else 2

                exact_key = f"{item['question']}||{item['answer']}"
                if exact_key in self.seen_exact:
                    self.log_action(sample_id, "deleted", "exact duplicate")
                    continue

                self.seen_exact.add(exact_key)
                item['hop_count'] = hop_count
                self.cleaned_data.append(item)
                self.log_action(sample_id, "retained", "valid sample")

            except Exception as e:
                self.log_action(sample_id, "skipped", f"error: {str(e)}")
                continue

        with open("musique_clean_logs.json", 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

        print(f"🧹 Cleaning completed: {len(data)} original -> {len(self.cleaned_data)} valid retained")
        return self.cleaned_data


def create_unique_title(item: dict) -> str:
    supporting_titles = []
    seen = set()
    sf = item.get('supporting_facts', [])
    for s in sf:
        title = s.get("title")
        if title and title not in seen:
            supporting_titles.append(title)
            seen.add(title)

    if not supporting_titles:
        return "UNKNOWN_TITLE"

    if len(supporting_titles) <= 3:
        return " and ".join(supporting_titles)
    else:
        return " and ".join(supporting_titles[:3]) + " et al"


def create_variable_name(title: str) -> str:
    name = re.sub(r'[^a-zA-Z0-9]', '_', title)
    name = re.sub(r'_+', '_', name).strip('_')
    if not name:
        name = "UNKNOWN"
    if name[0].isdigit():
        name = 'CONTEXT_' + name
    else:
        name = 'CONTEXT_' + name.upper()
    return name


def extract_context_text(item: dict) -> str:
    context_parts = []
    ctx = item.get('context', [])
    for doc in ctx:
        doc_title = doc.get("title", "")
        paragraph = doc.get("text", "")
        if doc_title and paragraph.strip():
            context_parts.append(f"# {doc_title}\n{paragraph.strip()}")

    return "\n\n".join(context_parts)


def generate_output_file(contexts: dict,
                         questions: dict,
                         metadata: dict,
                         output_path: str,
                         include_metadata: bool = True):
    """
    Generate output Python file in HotpotQA aligned format.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write('"""\n')
        f.write('Processed MuSiQue Dataset for TitanRAG Experiments\n')
        f.write('\n')
        f.write('This file contains processed contexts and question-answer pairs\n')
        f.write('extracted from the MuSiQue dataset.\n')
        f.write('\n')
        f.write('Note: MuSiQue is a multi-hop QA dataset requiring reasoning\n')
        f.write('across 20 paragraphs. Contexts combine supporting and distractor documents.\n')
        f.write('"""\n\n')

        # Contexts
        f.write('# ============================================================\n')
        f.write('# CONTEXTS - Cleaned and combined documents (20 paragraphs)\n')
        f.write('# ============================================================\n\n')

        for title, context in contexts.items():
            var_name = create_variable_name(title)

            if include_metadata and title in metadata:
                meta = metadata.get(title, {})
                hop_counts = list(set(meta.get('hop_counts', [2])))
                f.write(f'# Hop counts: {", ".join(map(str, hop_counts))}\n')
                f.write(f'# Number of questions: {len(questions[title])}\n')

            escaped_context = context.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
            f.write(f'{var_name} = """\n{escaped_context}\n"""\n\n')

        # Questions
        f.write('# ============================================================\n')
        f.write('# TEST_QUESTIONS - Question-Answer pairs\n')
        f.write('# ============================================================\n\n')

        f.write('TEST_QUESTIONS = {\n')
        for title, qa_pairs in questions.items():
            if not qa_pairs: continue
            escaped_title = title.replace('\\', '\\\\').replace('"', '\\"')
            f.write(f'    "{escaped_title}": [\n')
            for q, a in qa_pairs:
                escaped_q = q.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
                escaped_a = str(a).replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ')
                f.write(f'        ("{escaped_q}", "{escaped_a}"),\n')
            f.write('    ],\n')
        f.write('}\n\n')

        # Metadata
        if include_metadata:
            f.write('# ============================================================\n')
            f.write('# METADATA - Hop counts and IDs\n')
            f.write('# ============================================================\n\n')

            f.write('QUESTION_METADATA = {\n')
            for title, meta in metadata.items():
                escaped_title = title.replace('\\', '\\\\').replace('"', '\\"')
                f.write(f'    "{escaped_title}": {{\n')
                f.write(f'        "hop_counts": {list(set(meta.get("hop_counts", [])))},\n')
                f.write(f'        "ids": {meta.get("ids", [])},\n')
                f.write('    },\n')
            f.write('}\n\n')

        # Helper Functions
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
            f.write('    """Return metadata for questions (hop_counts, IDs)."""\n')
            f.write('    return QUESTION_METADATA\n')


def load_raw_data(input_path: str) -> List[Dict]:
    """Robustly load JSON or JSONL format"""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    if content.startswith('['):
        return json.loads(content)

    data = []
    for line in content.split('\n'):
        if line.strip():
            data.append(json.loads(line))
    return data


def main():
    # Configuration aligned with HotpotQA script
    RAW_FILE = "raw_musique.jsonl"
    OUTPUT_FILE = "processed_musique.py"

    INCLUDE_METADATA = True
    MAX_ITEMS = None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, RAW_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)

    print("=" * 60)
    print("MuSiQue Dataset Processor (20-paragraph multi-hop)")
    print("=" * 60)
    print(f"Metadata: {'Included' if INCLUDE_METADATA else 'Excluded'}")
    if MAX_ITEMS:
        print(f"Processing: First {MAX_ITEMS} items")
    print("=" * 60)

    # Check for both .jsonl and .json, auto-download if missing
    if not os.path.exists(input_path):
        alt_path = input_path.replace('.jsonl', '.json')
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            print(f"\n⚠️ Input file not found: {RAW_FILE}")
            download_musique_dataset(input_path)

    try:
        print("\n[1/3] Loading data...")
        raw_data = load_raw_data(input_path)

        if isinstance(raw_data, dict) and 'data' in raw_data:
            raw_data = raw_data['data']

        print(f"✅ Successfully loaded {len(raw_data)} QA samples")
    except Exception as e:
        print(f"\n❌ Failed to load data: {type(e).__name__} - {str(e)}")
        sys.exit(1)

    try:
        print("\n[2/3] Processing dataset...")
        cleaner = MuSiQueCleaner()
        cleaned_list = cleaner.process_dataset(raw_data, max_items=MAX_ITEMS)
    except Exception as e:
        print(f"\n❌ Cleaning error: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        sys.exit(1)

    if not cleaned_list:
        print("❌ Error: No valid data left after cleaning.")
        sys.exit(1)

    try:
        print("\n[3/3] Generating output file...")
        title_to_items = defaultdict(list)
        for item in cleaned_list:
            title = create_unique_title(item)
            title_to_items[title].append(item)

        contexts_dict = {}
        questions_dict = {}
        metadata_dict = {}

        for title, items in title_to_items.items():
            contexts_dict[title] = extract_context_text(items[0])
            questions_dict[title] = [(item['question'], item['answer']) for item in items]
            metadata_dict[title] = {
                'hop_counts': [item.get('hop_count', 2) for item in items],
                'ids': [item.get('id', 'unknown') for item in items]
            }

        generate_output_file(
            contexts_dict,
            questions_dict,
            metadata_dict,
            output_path,
            include_metadata=INCLUDE_METADATA
        )

        # Summary aligned with HotpotQA output
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  - Unique context groups: {len(contexts_dict)}")
        print(f"  - Total QA pairs: {sum(len(qa) for qa in questions_dict.values())}")
        print(f"  - Output file: {output_path}")

        # Show hop count distribution
        if metadata_dict:
            all_hops = []
            for meta in metadata_dict.values():
                all_hops.extend(meta['hop_counts'])

            from collections import Counter
            hop_counts = Counter(all_hops)

            print(f"\n  Hop Counts (Difficulty):")
            for hops, count in hop_counts.items():
                print(f"    - {hops} hops: {count}")

        print("=" * 60)

    except Exception as e:
        print(f"\n❌ File generation error: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()