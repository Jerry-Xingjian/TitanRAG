# 下载 Flan-T5-Large 模型和分词器到本地 model/flan-t5-large 目录
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import T5ForConditionalGeneration, T5Tokenizer


save_dir = os.path.join(os.path.dirname(__file__), '../model/flan-t5-large')
os.makedirs(save_dir, exist_ok=True)

print(f"下载模型和分词器到: {save_dir}")
T5ForConditionalGeneration.from_pretrained('google/flan-t5-large', cache_dir=save_dir)
T5Tokenizer.from_pretrained('google/flan-t5-large', cache_dir=save_dir)
print("下载完成！")
