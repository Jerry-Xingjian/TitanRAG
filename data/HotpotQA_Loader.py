import json
import os
import sys


def download_hotpot_if_missing():
    """下载 HotpotQA 训练集 (如果没有的话)"""
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
    filename = "hotpot_train_v1.1.json"

    if not os.path.exists(filename):
        print(f"📥 正在下载 {filename}...")
        # 使用 wget 或 curl，取决于环境，这里使用 python request 的简化版逻辑
        import urllib.request
        try:
            urllib.request.urlretrieve(url, filename)
            print("✅ 下载完成")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            print("请尝试手动运行: !wget " + url)
            sys.exit(1)
    return filename


def convert_to_essay_format(json_path, output_append_path, num_samples=5):
    """
    读取 HotpotQA 并将其追加到 sample_essays.py 中
    格式要求:
      variable_name = "context string"
      variable_name_questions = [("Question", "Answer")]
    """
    print(f"⚙️ 正在处理 HotpotQA 数据 (提取前 {num_samples} 条)...")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 筛选 bridge 类型的问题，通常更适合多跳推理测试
    bridge_questions = [item for item in data if item['type'] == 'bridge']
    selected_items = bridge_questions[:num_samples]

    new_content = "\n# " + "=" * 50 + "\n# HOTPOT QA ADAPTER DATA\n# " + "=" * 50 + "\n\n"

    generated_keys = []

    for idx, item in enumerate(selected_items):
        # 1. 构建 Context (将多个段落合并)
        # Hotpot 格式: context = [ ["Title", ["sent1", "sent2"]], ... ]
        full_text = []
        for title, sentences in item['context']:
            paragraph = "".join(sentences)
            full_text.append(f"--- {title} ---\n{paragraph}")

        combined_text = "\n\n".join(full_text)

        # 转义处理，防止 Python 字符串断裂
        safe_text = combined_text.replace('"""', '\\"\\"\\"')
        safe_q = item['question'].replace('"', '\\"')
        safe_a = item['answer'].replace('"', '\\"')

        # 2. 生成变量名
        key = f"hotpot_{idx}"
        generated_keys.append(key)

        # 3. 写入 Python 代码格式
        new_content += f'{key} = """{safe_text}"""\n\n'
        new_content += f'{key}_questions = [\n    ("{safe_q}", "{safe_a}")\n]\n\n'

    # 将内容追加到 data/sample_essays.py
    # 这样 main.py 导入时就能看到这些新变量
    if os.path.exists(output_append_path):
        with open(output_append_path, 'a', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ 已成功将 {len(generated_keys)} 条 Hotpot 数据追加到 {output_append_path}")
        print(f"可用名称: {', '.join(generated_keys)}")
    else:
        print(f"❌ 错误: 找不到 {output_append_path}")


if __name__ == "__main__":
    json_file = download_hotpot_if_missing()
    # 假设脚本在根目录运行，目标文件在 data/ 下
    target_file = "data/sample_essays.py"

    # 如果 data 目录不存在（例如在 colab 根目录），尝试直接找文件名
    if not os.path.exists(target_file) and os.path.exists("sample_essays.py"):
        target_file = "sample_essays.py"

    convert_to_essay_format(json_file, target_file)