import json
import random

def main():
    path = 'data/chunk_feature_label.json'
    output_path = 'data/chunk_feature_label_balanced.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    retrievable = [item for item in data if item['is_retrievable']]
    not_retrievable = [item for item in data if not item['is_retrievable']]
    n = min(len(retrievable), len(not_retrievable))
    retrievable_sample = random.sample(retrievable, n)
    not_retrievable_sample = random.sample(not_retrievable, n)
    balanced_data = retrievable_sample + not_retrievable_sample
    random.shuffle(balanced_data)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    print(f'已生成平衡数据集: {output_path} (每类样本数: {n})')

if __name__ == '__main__':
    main()
