import os
import argparse
import json
import re
import sys

def generate_answer_label_dataset(data_path, output_path):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from main import TitanRAG
    # 初始化TitanRAG模型（此处需根据你的实际模型加载方式调整）
    titan_model = None  # TODO: 替换为实际模型加载
    rag = TitanRAG(titan_model)

    with open(data_path, "r", encoding="utf-8") as f:
        squad_data = json.load(f)
    answer_label_data = []
    for article in squad_data.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                # 用TitanRAG推理
                # 这里假设rag.query_with_context能返回答案字符串，否则请补充实际推理逻辑
                rag_answer = None
                try:
                    # 伪代码：rag_answer = rag.query_with_context(question, [context])
                    rag_answer = None  # TODO: 替换为实际推理结果
                except Exception:
                    rag_answer = None
                if rag_answer and isinstance(rag_answer, str) and rag_answer.strip():
                    answer_label_data.append({"text": rag_answer.strip(), "label": "可检索"})
                else:
                    answers = qa.get("answers", [])
                    if answers:
                        gold = answers[0].get("text", "")
                        sents = re.split(r'[。！？!?.]', context)
                        found = None
                        for sent in sents:
                            if gold in sent:
                                found = sent.strip()
                                break
                        if found:
                            answer_label_data.append({"text": found, "label": "不可检索"})
                        else:
                            answer_label_data.append({"text": gold, "label": "不可检索"})
                    else:
                        continue
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answer_label_data, f, ensure_ascii=False, indent=2)
    print(f"已保存: {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answer label dataset for TitanRAG")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data (SQuAD format)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output label dataset")
    args = parser.parse_args()

    generate_answer_label_dataset(args.data_path, args.output_path)