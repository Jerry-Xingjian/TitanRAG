import json

def main():
    path = 'data/chunk_feature_label.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    c1 = sum(1 for i in data if i['is_retrievable'])
    c0 = sum(1 for i in data if not i['is_retrievable'])
    print(f'可检索(1): {c1}')
    print(f'不可检索(0): {c0}')
    print(f'总数: {len(data)}')

if __name__ == '__main__':
    main()
