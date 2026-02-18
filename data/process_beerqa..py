"""
Script to strictly clean, and process local BeerQA dataset.
Fully compatible with both official BeerQA and HotpotQA-style format.
"""

import json
import re
import os
import sys
import io
import traceback
from collections import defaultdict
from typing import List, Dict

# Fix Windows console encoding issue
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================================

class BeerQACleaner:
    def __init__(self):
        self.cleaned_data = []
        self.seen_exact = set()
        self.logs = []

        # Answer normalization
        self.answer_norm_dict = {
            "USA": "United States",
            "US": "United States",
            "U.S.": "United States",
            "UK": "United Kingdom"
        }
        self.blocklist = ["bad_word1", "bad_word2", "forbidden_word"]

    def log_action(self, sample_id: str, action: str, reason: str):
        """Log cleaning actions"""
        self.logs.append({"sample_id": sample_id, "action": action, "reason": reason})

    def normalize_text(self, text: str) -> str:
        """Text normalization"""
        if not isinstance(text, str):
            return ""
        text = text.strip().replace('\n', ' ').replace('\r', '').replace('\t', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text

    def check_compliance(self, text: str) -> bool:
        """Compliance check"""
        text_lower = text.lower()
        for bad_word in self.blocklist:
            if bad_word.lower() in text_lower:
                return False
        return True

    def process_dataset(self, data: List[Dict]) -> List[Dict]:
        """Execute cleaning, compatible with both list/dict format"""
        self.logs = []
        if not isinstance(data, list):
            raise ValueError(f"Dataset format error: expected list, got {type(data)}")

        for idx, item in enumerate(data):
            sample_id = item.get('id', item.get('_id', f"beerqa_{idx}"))
            try:
                q = item.get('question', item.get('query', ''))
                a = item.get('answer', item.get('answers', ''))
                sf = item.get('supporting_facts', item.get('supporting_paragraphs', []))
                ctx = item.get('context', item.get('paragraphs', []))
                num_hops = item.get('num_hops', item.get('hop_count', item.get('level', 1)))

                if not q or not a:
                    self.log_action(sample_id, "deleted", "question/answer is empty")
                    continue
                if len(q) < 3:
                    self.log_action(sample_id, "deleted", "question is too short (<3 chars)")
                    continue

                # Text normalization
                item['question'] = self.normalize_text(q)
                item['answer'] = self.normalize_text(str(a))
                if isinstance(item['answer'], str):
                    item['answer'] = self.answer_norm_dict.get(item['answer'], item['answer'])

                # Compliance check
                if not self.check_compliance(item['question'] + str(item['answer'])):
                    self.log_action(sample_id, "deleted", "triggered blocklist")
                    continue

                # Exact duplicate check
                sf_str = json.dumps(sf, ensure_ascii=False)
                exact_key = f"{item['question']}||{item['answer']}||{sf_str}"
                if exact_key in self.seen_exact:
                    self.log_action(sample_id, "deleted", "exact duplicate")
                    continue

                # ========================================================================
                # 修复逻辑: 提取有效 title, 如果 supporting_facts 为空，则回退使用所有 context 标题
                # ========================================================================
                sf_titles = set()
                valid_sf = []
                for s in sf:
                    if isinstance(s, list) and len(s) >= 1:
                        title = s[0]
                        sf_titles.add(title)
                        valid_sf.append(s)
                    elif isinstance(s, dict) and "title" in s:
                        title = s["title"]
                        sf_titles.add(title)
                        valid_sf.append(s)


                if not sf_titles:
                    for doc in ctx:
                        if isinstance(doc, list) and len(doc) >= 1:
                            title = doc[0]
                            sf_titles.add(title)
                            valid_sf.append([title, 0])
                        elif isinstance(doc, dict) and "title" in doc:
                            title = doc["title"]
                            sf_titles.add(title)
                            valid_sf.append({"title": title, "sent_id": 0})

                item['supporting_facts'] = valid_sf

                # 过滤 Context
                filtered_context = []
                for doc in ctx:
                    if isinstance(doc, list) and len(doc) >= 1:
                        doc_title = doc[0]
                        if doc_title in sf_titles:
                            filtered_context.append(doc)
                    elif isinstance(doc, dict) and "title" in doc:
                        doc_title = doc["title"]
                        if doc_title in sf_titles:
                            filtered_context.append(doc)
                item['context'] = filtered_context


                self.seen_exact.add(exact_key)
                self.cleaned_data.append(item)
                self.log_action(sample_id, "retained", "valid sample")

            except Exception as e:
                self.log_action(sample_id, "skipped", f"processing error: {type(e).__name__} - {str(e)}")
                continue

        with open("beerqa_clean_logs.json", 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

        print(f"🧹 Cleaning completed: {len(data)} original -> {len(self.cleaned_data)} valid retained")
        return self.cleaned_data


# ============================================================================
# 2. Format Conversion Module
# ============================================================================
def create_unique_title(item: dict) -> str:
    """Create unique title from supporting facts"""
    supporting_titles = []
    seen = set()
    sf = item.get('supporting_facts', [])
    for s in sf:
        title = None
        if isinstance(s, list) and len(s) >= 1:
            title = s[0]
        elif isinstance(s, dict) and "title" in s:
            title = s["title"]
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
    """Create valid Python variable name"""
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
    """Extract context text robustly"""
    context_parts = []
    ctx = item.get('context', [])
    for doc in ctx:
        doc_title = ""
        sentences = None
        if isinstance(doc, list) and len(doc) >= 2:
            doc_title = doc[0]
            sentences = doc[1]
        elif isinstance(doc, dict) and "title" in doc:
            doc_title = doc["title"]
            sentences = doc.get("sentences", doc.get("text", ""))  # 兼容多种字段

        if doc_title and sentences is not None:
            if isinstance(sentences, list):
                paragraph = " ".join([str(s) for s in sentences])
            else:
                paragraph = str(sentences)

            if paragraph.strip():
                context_parts.append(f"# {doc_title}\n{paragraph}")
    return "\n\n".join(context_parts)


def generate_output_file(contexts: dict, questions: dict, metadata: dict, output_path: str):
    """Generate HotpotQA-compatible output file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('"""\nProcessed BeerQA Dataset for TitanRAG Experiments\n"""\n\n')

        f.write('# ============================================================\n')
        f.write('# CONTEXTS - Cleaned and combined supporting documents\n')
        f.write('# ============================================================\n\n')

        for title, context in contexts.items():
            var_name = create_variable_name(title)
            meta = metadata.get(title, {})

            q_types = set(meta.get('type', ['unknown']))
            hop_counts = set(meta.get('num_hops', [1]))
            f.write(f'# Question types: {", ".join(q_types)}\n')
            f.write(f'# Hop counts: {", ".join(map(str, hop_counts))}\n')

            escaped_context = context.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
            f.write(f'{var_name} = """\n{escaped_context}\n"""\n\n')

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

        f.write('# ============================================================\n')
        f.write('# METADATA - Question types and hop counts\n')
        f.write('# ============================================================\n\n')

        f.write('QUESTION_METADATA = {\n')
        for title, meta in metadata.items():
            escaped_title = title.replace('\\', '\\\\').replace('"', '\\"')
            f.write(f'    "{escaped_title}": {{\n')
            # 修复：移除 list(set(...)) 以保证列表长度与 questions 数量一一对应
            f.write(f'        "types": {meta.get("type", [])},\n')
            f.write(f'        "hop_counts": {meta.get("num_hops", [])},\n')
            f.write(f'        "ids": {meta.get("ids", [])},\n')
            f.write('    },\n')
        f.write('}\n\n')

        f.write('# ============================================================\n')
        f.write('# HELPER FUNCTIONS\n')
        f.write('# ============================================================\n\n')

        f.write('def get_all_contexts():\n')
        f.write('    return {\n')
        for title in contexts.keys():
            var_name = create_variable_name(title)
            escaped_title = title.replace('\\', '\\\\').replace('"', '\\"')
            f.write(f'        "{escaped_title}": {var_name},\n')
        f.write('    }\n\n')

        f.write('def get_test_questions():\n')
        f.write('    return TEST_QUESTIONS\n\n')

        f.write('def get_question_metadata():\n')
        f.write('    return QUESTION_METADATA\n')


# ============================================================================
# 3. Main Execution Flow
# ============================================================================
def main():
    RAW_FILE = "raw_beerqa.json"
    OUTPUT_FILE = "processed_beerqa.py"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, RAW_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)

    print("=" * 60)
    print("BeerQA Dataset Local Processor (Full Format Compatible)")
    print("=" * 60)

    # 检查本地文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at {input_path}")
        print(f"   Please ensure '{RAW_FILE}' is placed in the same directory as this script.")
        sys.exit(1)

    # 加载数据集
    try:
        print("\n[1/3] Loading local BeerQA dataset...")
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)

        if isinstance(raw_json, dict):
            raw_data = raw_json.get('data', raw_json.get('questions', []))
        elif isinstance(raw_json, list):
            raw_data = raw_json
        else:
            raise ValueError(f"Unsupported JSON format: root is {type(raw_json)}")

        print(f"✅ Successfully loaded {len(raw_data)} QA samples")
    except Exception as e:
        print(f"\n❌ Failed to load data: {type(e).__name__} - {str(e)}")
        sys.exit(1)

    # 数据清洗
    try:
        print("\n[2/3] Running strict data cleaning...")
        cleaner = BeerQACleaner()
        cleaned_list = cleaner.process_dataset(raw_data)
    except Exception as e:
        print(f"\n❌ Cleaning error: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        sys.exit(1)

    if not cleaned_list:
        print("❌ Error: No valid data left after cleaning. Check 'beerqa_clean_logs.json'.")
        sys.exit(1)

    # 生成文件
    try:
        print("\n[3/3] Generating baseline-compatible output file...")
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
                'type': [item.get('type', 'unknown') for item in items],
                'num_hops': [item.get('num_hops', item.get('hop_count', 1)) for item in items],
                'ids': [item.get('id', item.get('_id', 'unknown')) for item in items]
            }

        generate_output_file(contexts_dict, questions_dict, metadata_dict, output_path)
        print(f"\n✅ All completed! Output file: {output_path}")
    except Exception as e:
        print(f"\n❌ File generation error: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()