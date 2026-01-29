# TitanRAG 研究路线图

## 研究方向

### 方向 A：Adaptive Memory-RAG (推荐)
**核心创新**：智能决定何时依赖 Memory vs RAG

| 模式 | 适用场景 |
|------|---------|
| titans_only | 推理/概念性问题 |
| hybrid | 事实/精确性问题 |
| **Adaptive** | 自动选择最优策略 |

**实验设计**：
- Baseline: Pure RAG, titans_only, hybrid (固定)
- 你的方法: HybridDecider (自适应)
- 指标: Accuracy, F1, 决策准确率

---

### 方向 B：Continual Learning QA
**核心创新**：系统在使用中持续学习

**实验设计**：
- 设计 session-based 问答场景
- 测量 Q1→Q10 准确率提升曲线
- 对比有无 Online Learning

---

### 方向 C：Titans + Tool Augmentation
**核心创新**：Memory 找数据 + 工具做计算

**适用场景**：FinQA 等需要数值计算的任务

---

## 数据集资源

### 金融/数值推理
| 数据集 | 链接 | 特点 |
|--------|------|------|
| FinQA | [GitHub](https://github.com/czyssrs/FinQA) | 金融报表 + 计算 |
| TAT-QA | [GitHub](https://github.com/NExTplusplus/TAT-QA) | 表格 + 文本 |
| ConvFinQA | [GitHub](https://github.com/czyssrs/ConvFinQA) | 多轮金融对话 |

### 长文档问答
| 数据集 | 链接 | 特点 |
|--------|------|------|
| NarrativeQA | [HuggingFace](https://huggingface.co/datasets/narrativeqa) | 书籍摘要 |
| QuALITY | [GitHub](https://github.com/nyu-mll/quality) | 长文档多选 |
| SCROLLS | [HuggingFace](https://huggingface.co/datasets/tau/scrolls) | 多任务长文本 |

### 多跳推理 (推荐)
| 数据集 | 链接 | 特点 |
|--------|------|------|
| HotpotQA | [HuggingFace](https://huggingface.co/datasets/hotpot_qa) | 多文档推理 |
| MuSiQue | [GitHub](https://github.com/StonyBrookNLP/musique) | 多步推理 |

### 通用 QA
| 数据集 | 链接 | 特点 |
|--------|------|------|
| SQuAD 2.0 | [HuggingFace](https://huggingface.co/datasets/squad_v2) | 经典阅读理解 |
| Natural Questions | [Google](https://ai.google.com/research/NaturalQuestions) | 真实搜索 |
| TriviaQA | [HuggingFace](https://huggingface.co/datasets/trivia_qa) | 知识问答 |

---

## 快速开始

```python
from datasets import load_dataset

# 加载 HotpotQA (推荐)
ds = load_dataset("hotpot_qa", "fullwiki")

# 加载 SQuAD 2.0
ds = load_dataset("squad_v2")

# 加载 NarrativeQA
ds = load_dataset("narrativeqa")
```

---

## 下一步行动

1. [ ] 选择一个数据集 (建议 HotpotQA)
2. [ ] 跑 Baseline (Pure RAG, titans_only, hybrid)
3. [ ] 对比你的 Adaptive 方法
4. [ ] 消融实验 (Memory 深度、决策权重等)
