"""
采集决策器训练数据：多次运行 eval_three_modes.py 的核心逻辑，保存特征和标签。
特征：retr_score, mem_sim, surprise
标签：context loss < mem loss（二分类）
输出：decision_maker_train.csv
"""
import numpy as np
import csv
from eval_three_modes import run_numpy_sim


with open('decision_maker_train.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['retr_score', 'mem_sim', 'surprise', 'label', 'seed', 'digest_steps', 'num_docs'])
    for seed in range(100, 400):
        for digest_steps in [50, 100, 200, 400, 800]:
            for num_docs in [5, 10, 20, 40]:
                class Args: pass
                args = Args()
                args.num_docs = num_docs
                args.doc_len = 32
                args.dim = 64
                args.digest_steps = digest_steps
                args.seed = seed
                res = run_numpy_sim(args)
                context_loss = res['mode_b']['loss']
                mem_loss = res['mode_a']['loss']
                c = res['mode_c']
                label = int(context_loss < mem_loss)
                writer.writerow([c['retr_score'], c['mem_sim'], c['surprise'], label, seed, digest_steps, num_docs])
print('采集完成，已保存为 decision_maker_train.csv')
