"""
用 sample_essays.py 的问答数据，模拟 TitanRAG 推理，采集决策器训练数据。
特征：retr_score, mem_sim, surprise, gate
输出：decision_maker_essays.csv
"""

import csv
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from data.sample_essays import get_all_essays, get_test_questions
from eval_three_modes import run_numpy_sim

random.seed(42)

with open('decision_maker_essays.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['topic', 'question', 'retr_score', 'mem_sim', 'surprise', 'gate'])
    essays = get_all_essays()
    questions = get_test_questions()
    for topic, qa_list in questions.items():
        for q, _ in qa_list:
            # 用每个问句作为 query，知识库为该主题 essay
            class Args: pass
            args = Args()
            args.num_docs = 20
            args.doc_len = 32
            args.dim = 64
            args.digest_steps = 200
            # 用 hash(q) 作为 seed 保证每个问句唯一性
            args.seed = abs(hash(q)) % (2**31)
            res = run_numpy_sim(args)
            c = res['mode_c']
            writer.writerow([topic, q, c['retr_score'], c['mem_sim'], c['surprise'], c['gate']])
print('采集完成，已保存为 decision_maker_essays.csv')
