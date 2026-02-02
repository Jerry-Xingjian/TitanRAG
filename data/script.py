import json
import re
from typing import Dict, List, Tuple, Any


def load_squad_data(file_path: str) -> Dict[str, Any]:
    """
    加载SQuAD2.0数据集（JSON格式）
    Args:
        file_path: SQuAD2.0数据集文件路径（train-v2.0.json/dev-v2.0.json）
    Returns:
        解析后的JSON字典
    """
    with open(file_path, "r", encoding="utf-8") as f:
        squad_data = json.load(f)
    return squad_data


def merge_context_by_title(squad_data: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, List[Dict]]]:
    """
    按title合并context，并保留原始问题-答案对
    Args:
        squad_data: 加载后的SQuAD数据集
    Returns:
        title_context_map: {title: 合并后的完整context}
        title_qas_map: {title: 该title下所有qas数据}
    """
    title_context_map = {}  # 存储每个title对应的合并后context
    title_qas_map = {}  # 存储每个title对应的所有问题-答案对

    for data_item in squad_data["data"]:
        title = data_item["title"].strip()
        paragraphs = data_item["paragraphs"]

        # 合并当前title下的所有context（去重+分段，保持可读性）
        merged_context = []
        for para in paragraphs:
            context = para["context"].strip()
            if context not in merged_context:  # 去重避免重复文本
                merged_context.append(context)

        # 拼接成类长文本格式（分段+换行）
        final_context = "\n\n".join(merged_context)
        title_context_map[title] = final_context

        # 收集当前title下的所有qas（问题-答案对）
        all_qas = []
        for para in paragraphs:
            all_qas.extend(para["qas"])
        title_qas_map[title] = all_qas

    return title_context_map, title_qas_map


def validate_answer_in_context(answer_text: str, context: str) -> bool:
    """
    验证答案是否存在于合并后的context中（兼容部分匹配/大小写忽略）
    Args:
        answer_text: 答案文本
        context: 合并后的完整context
    Returns:
        答案是否在context中（布尔值）
    """
    if not answer_text:  # 处理不可回答问题
        return True

    # 模糊匹配（忽略大小写、标点、多余空格）
    clean_answer = re.sub(r"[^\w\s]", "", answer_text.lower()).strip()
    clean_context = re.sub(r"[^\w\s]", "", context.lower()).strip()

    # 精确子串匹配（核心逻辑：确保答案确实来自context）
    return clean_answer in clean_context


def process_squad_for_titan_model(
        squad_file_path: str,
        output_json_path: str = "processed_squad_titan.json"
) -> Dict[str, Any]:
    """
    处理SQuAD2.0数据集，输出适配Titan大模型的格式
    格式参考sample_essays.py的TEST_QUESTIONS结构
    Args:
        squad_file_path: 原始SQuAD2.0文件路径
        output_json_path: 处理后的数据输出路径
    Returns:
        处理后的完整数据集（字典格式）
    """
    # 1. 加载原始数据
    squad_data = load_squad_data(squad_file_path)
    print(f"✅ 成功加载SQuAD2.0数据集，共{len(squad_data['data'])}个title")

    # 2. 按title合并context
    title_context_map, title_qas_map = merge_context_by_title(squad_data)
    print(f"✅ 成功合并context，共{len(title_context_map)}个唯一title")

    # 3. 构建模型训练数据（参考sample_essays.py的TEST_QUESTIONS格式）
    titan_training_data = {
        "contexts": {},  # {title: 合并后的context}
        "questions": {}  # {title: [(question, answer, is_impossible), ...]}
    }

    invalid_qa_count = 0  # 统计答案不在context中的无效QA
    total_qa_count = 0  # 统计总QA数

    for title, context in title_context_map.items():
        titan_training_data["contexts"][title] = context
        qas_list = title_qas_map.get(title, [])
        title_questions = []

        for qa in qas_list:
            total_qa_count += 1
            question = qa["question"].strip()
            is_impossible = qa.get("is_impossible", False)
            answer_text = ""

            # 处理可回答问题：提取答案文本
            if not is_impossible and qa.get("answers"):
                answer_text = qa["answers"][0]["text"].strip()  # 取第一个答案（SQuAD可能有多个）

                # 验证答案是否在context中
                if not validate_answer_in_context(answer_text, context):
                    invalid_qa_count += 1
                    continue  # 过滤答案不在context中的QA

            # 加入训练数据（格式：(问题, 答案, 是否不可回答)）
            title_questions.append((question, answer_text, is_impossible))

        titan_training_data["questions"][title] = title_questions

    # 4. 输出处理后的数据
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(titan_training_data, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print(f"✅ 数据处理完成！输出路径：{output_json_path}")
    print(f"📊 统计信息：")
    print(f"   - 总QA数：{total_qa_count}")
    print(f"   - 无效QA数（答案不在context）：{invalid_qa_count}")
    print(f"   - 有效QA数：{total_qa_count - invalid_qa_count}")
    print(f"   - 最终保留title数：{len(titan_training_data['contexts'])}")

    return titan_training_data


def convert_to_sample_essay_format(titan_data: Dict[str, Any], output_path: str = "squad_sample_format.py"):
    """
    将处理后的数据转换为sample_essays.py的代码格式（可选）
    方便直接集成到现有代码库中
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('"""Processed SQuAD2.0 Data for TitanRAG Experiments"""')
        f.write("\n\n")

        # 写入合并后的context
        f.write("# Merged Contexts by Title\n")
        for title, context in titan_data["contexts"].items():
            # 处理title的命名（转为合法变量名）
            var_name = "CONTEXT_" + re.sub(r"[^A-Z0-9]", "_", title.upper())
            f.write(f"{var_name} = '''{context}'''\n\n")

        # 写入问题-答案对（类似TEST_QUESTIONS）
        f.write("# Test Questions for Titan Model\n")
        f.write("TEST_QUESTIONS_SQUAD = {\n")
        for title, questions in titan_data["questions"].items():
            # 转义引号，格式化问题列表
            f.write(f'    "{title}": [\n')
            for q, a, is_impossible in questions:
                # 处理不可回答问题的答案标注
                answer_str = a if not is_impossible else "NO_ANSWER"
                f.write(f'        ("{q}", "{answer_str}"),\n')
            f.write("    ],\n")
        f.write("}\n")

    print(f"✅ 已转换为sample_essay格式，输出路径：{output_path}")


if __name__ == "__main__":
    # -------------------------- 配置参数 --------------------------
    SQUAD_FILE_PATH = "train-v2.0.json"  # 替换为你的SQuAD2.0文件路径
    OUTPUT_JSON_PATH = "processed_squad_titan.json"
    OUTPUT_SAMPLE_FORMAT_PATH = "squad_sample_format.py"








