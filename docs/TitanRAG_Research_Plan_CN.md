# TitanRAG 研究计划

## 📋 概述

**项目目标**：探索将 Titans 的 Test-Time Training (TTT) 机制应用于 RAG 系统的可行性和效果。

---

## 🆚 原始 Titans vs Hybrid Titan 对比

### 核心相同点

| 特性 | 原始 Titans | Hybrid Titan |
|------|-------------|--------------|
| Test-Time Training | ✅ | ✅ |
| `forward_with_update` | ✅ | ✅ |
| 推理时更新权重 | ✅ | ✅ |

### 关键区别

| 维度 | 原始 Titans | Hybrid Titan |
|------|-------------|--------------|
| **学习内容** | Token 序列 (自回归) | 语义知识 (Question→Answer) |
| **输入来源** | 仅 Context Window | Context + **外部检索** |
| **检索能力** | 无 | Ensemble Fusion |
| **任务类型** | 语言建模 | 问答 / RAG |
| **需要 LM Head** | ✅ | ❌ (可选) |

### 架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│ 原始 Titans:                                                    │
│                                                                 │
│   "The cat sat on..." → Memory.update(token_sequence)          │
│                         ↓                                       │
│   预测下一个 Token: "the" (需要 LM Head)                        │
├─────────────────────────────────────────────────────────────────┤
│ Hybrid Titan:                                                   │
│                                                                 │
│   Essay文档 → Memory.update(semantic_embeddings)                │
│                         ↓                                       │
│   Question → Memory + Retriever → 检索相关段落                  │
│              (Ensemble Fusion: Keyword + Memory + Embedding)    │
└─────────────────────────────────────────────────────────────────┘
```

### Hybrid Titan 的扩展点

1. **应用场景扩展**：从"语言建模"到"知识检索"
2. **与 Retriever 结合**：Ensemble Fusion (Keyword + Memory + Embedding)
3. **可解释性**：能看到检索了哪些文档段落

---

## ✅ 已完成的实验

### Phase 1: 基础架构

| 组件 | 状态 | 说明 |
|------|------|------|
| `DeepMemoryModule` | ✅ | 实现 TTT 的核心模块 |
| `TitanMAG` | ✅ | Memory-As-Gate 融合 |
| `TitanRAG` | ✅ | Memory + Attention + RAG |
| Semantic Embedder | ✅ | sentence-transformers (all-MiniLM-L6-v2) |

### Phase 2: Essay RAG Demo

#### 2.1 Query Demo (检索能力测试)

**目的**: 测试 Memory 的语义检索能力

**流程**:
```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Digestion: Essay → embedder → LTM.forward_with_update()     │
│    (50 epochs, 权重更新)                                        │
├─────────────────────────────────────────────────────────────────┤
│ 2. Query: Question → embedder → LTM.forward_no_update()        │
│    (只读取，不更新)                                              │
├─────────────────────────────────────────────────────────────────┤
│ 3. Retrieval: cosine_similarity(Memory输出, Essay各行嵌入)     │
│    → 找出 Top-K 最相似的行                                      │
├─────────────────────────────────────────────────────────────────┤
│ 4. Online Learning: LTM.forward_with_update(Question, Answer)  │
│    → 展示 BEFORE/AFTER 相似度变化                               │
└─────────────────────────────────────────────────────────────────┘
```

**评估指标**: Memory Top-1 是否为正确答案所在行

---

#### 2.2 Hybrid RAG Demo (融合检索测试)

**目的**: 测试 Memory 背景 + Retriever 精确上下文 的融合效果

**流程**:
```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Ensemble Fusion Retrieval (三方法融合):                      │
│    ├─ Keyword: 关键词匹配分数                                   │
│    ├─ Memory:  LTM 语义相似度                                   │
│    └─ Embedding: 直接问题-行嵌入相似度                          │
│    → 加权融合 (0.3:0.4:0.3) → 选出 best_line_idx               │
├─────────────────────────────────────────────────────────────────┤
│ 2. Multi-Line Context Window:                                   │
│    [prev_line] [CENTER×2] [next_line] (中心行 2x 权重)          │
├─────────────────────────────────────────────────────────────────┤
│ 3. Hybrid Input: Context + Question → TitanRAG                  │
│    → 模型同时看到 Memory(背景) + Context(精确)                  │
├─────────────────────────────────────────────────────────────────┤
│ 4. 评估: cosine_similarity(TitanRAG输出, Expected Answer嵌入)   │
│    对比: Hybrid RAG vs Memory-only → 计算提升比例              │
└─────────────────────────────────────────────────────────────────┘
```

**评估指标**: Hybrid RAG 相对于 Memory-only 的提升百分比

---

#### 量化结果 (最新运行)

**Memory Retention**: Loss Improvement **92-94%**

**Query Demo (Memory Top-1 正确率)**:

| Essay | Q1 | Q2 | Q3 | Q4 | 正确率 |
|-------|----|----|----|----|--------|
| Climate | ✅ | ✅ | ✅ | ✅ | 100% |
| AI | ✅ | ✅ | ✅ | ✅ | 100% |
| Space | ⚠️ | ✅ | ✅ | ✅ | 75% |

**Hybrid RAG (vs Memory-only 提升)**:

| Essay | Q1 | Q2 | Q3 | Q4 |
|-------|----|----|----|----|
| Climate | +10.5% | +4.0% | +24.3% | +10.3% |
| AI | +7.1% | +8.2% | +5.0% | +12.8% |
| Space | +3.9% | +15.7% | +0.8% | +3.6% |

---

## 📊 当前系统分析

### 优势
- ✅ TTT 机制有效：Online Learning 展示 0.2-1% 即时改进
- ✅ Ensemble Fusion 更稳定：三种方法通常达成共识
- ✅ Hybrid RAG 普遍优于 Memory-only

### 局限性
- ⚠️ 输出相似度较低 (0.1-0.3)：TitanRAG 输出不在原始嵌入空间
- ⚠️ 仅评估检索能力，无生成能力
- ⚠️ 数据集规模小（3篇 Essay × 4 问题）

---

## 🚀 研究方向

### Direction A: Hybrid Titan (外部 LLM 生成) ✅ 已实现

**状态**: ✅ **已完成** (2026-01-23)

**架构**: TitanRAG 检索 + Flan-T5-Large 生成

```
Question → TitanRAG (Ensemble Fusion) → Context → Flan-T5-Large → Answer
                ↑                              ↓
         Online Learning ←──── (Q, Expected_A) ←────┘
```

**实现特性**:

| 组件 | 配置 | 说明 |
|------|------|------|
| **生成模型** | `google/flan-t5-large` | 783M 参数, 指令优化 |
| **检索方式** | Ensemble Fusion | Keyword(0.2) + Memory(0.5) + Embedding(0.3) |
| **上下文切分** | 段落级 | `essay.split('\n\n')` 保持语义完整 |
| **Online Learning** | ✅ 启用 | `forward_with_update(Q, A)` 即时更新 |
| **Top-K** | 5 | 检索 5 个最相关段落 |

**文件**: `experiments/hybrid_titan_demo.py`

**运行方式**:
```bash
python experiments/hybrid_titan_demo.py --essay climate --epochs 50 --topk 5
python experiments/hybrid_titan_demo.py --essay ai --epochs 50 --topk 5
python experiments/hybrid_titan_demo.py --essay space --epochs 50 --topk 5
```

**优点**: 
- ✅ 实现简单，可用现成 LLM
- ✅ Flan-T5 指令遵循能力强
- ✅ Online Learning 增强检索质量

**待优化**:
- ⚠️ 生成质量受限于上下文检索准确度
- ⚠️ 特定事实类问题需进一步调优

---

### Direction B: Native LM Head (完整 Titans 架构) ⭐ 待实现

**核心洞察**: Titans 论文将 Memory 嵌入完整 Transformer，通过 LM Head 生成

**架构**:
```
┌─────────────────────────────────────────┐
│  Input Tokens                           │
│       ↓                                 │
│  [Attention + Memory (TTT)] × N Layers  │
│       ↓                                 │
│  LM Head (Linear → Vocab)  ← 生成能力   │
│       ↓                                 │
│  Next Token Prediction                  │
└─────────────────────────────────────────┘
```

**实现**:
```python
class TitanWithLMHead(nn.Module):
    def __init__(self, titan_core, vocab_size=50257):
        self.titan = titan_core
        self.lm_head = nn.Linear(dim, vocab_size)
    
    def forward(self, input_ids):
        hidden = self.titan(input_ids)  # Memory + Attention
        logits = self.lm_head(hidden)   # → Vocab
        return logits
    
    def generate(self, prompt, max_len=50):
        for _ in range(max_len):
            logits = self.forward(prompt)
            next_token = logits[:, -1].argmax(-1)
            prompt = torch.cat([prompt, next_token], dim=-1)
        return prompt
```

**评估**: Perplexity, Next Token Accuracy

---

### Direction C: 标准 Benchmark 评估

| 任务 | 数据集 | 指标 |
|------|--------|------|
| 语言建模 | WikiText-103 | Perplexity |
| 问答 | Natural Questions | EM/F1 |
| 长文档 | SCROLLS | ROUGE-L |

---

## 📅 时间线更新

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Phase 1** | 基础架构 (TitanRAG, TitanMAG) | ✅ 完成 |
| **Phase 2** | Essay RAG Demo (Query + Hybrid) | ✅ 完成 |
| **Phase 3a** | Hybrid Titan (Flan-T5-Large) | ✅ 完成 |
| **Phase 3b** | Native LM Head | 🔲 待实现 |
| **Phase 4** | Benchmark 评估 | 🔲 待实现 |
| **Phase 5** | 消融实验 + 可视化 | 🔲 待实现 |

---

## 🔮 挑战与解决方案 (Challenges & Solutions)

### 1. 冲突事实 (Conflicting Truth)
*   **问题**: Neural Memory (幻觉/旧知识) 与 Retrieved Context (新事实) 冲突。
*   **解决方案**: **Confidence Gating (置信度门控)**.
    *   在 `TitanMAG` 中，不要只让模型自己学习 Gate。
    *   **Force Override**: 如果检索到的文档与 Memory 差异过大（High Surprise），这通常意味着"新知识"。在这种情况下，显式增加 Context 分支的权重，压制 Memory 分支。
    *   `Gate = sigmoid(Learning_Rate * Surprise + Bias)`

### 2. 双重训练开销 (Double Training Overhead)
*   **问题**: TTT 导致推理延迟高（每篇文档都要反向传播）。
*   **解决方案**: **Selective Digestion (选择性消化)**.
    *   不要对所有检索到的 Top-K 文档都运行 `forward_with_update`。
    *   只对 **Relevance Score > 0.8** 的高置信度文档进行 TTT。
    *   或者：只对 User 明确标记为"有用"的文档进行 TTT。

### 3. 噪声毒化 (Noise Poisoning)
*   **问题**: 错误文档污染模型权重。
*   **解决方案**: **Episodic Reset (重置机制)**.
    *   **Session-based weights**: 每次新对话开始时，重置 LTM 到初始状态，防止上一次对话的错误信息残留。
    *   **Rollback**: 如果用户反馈"回答错误"，执行 `rollback()` 操作，撤销最近一次的梯度更新（需要保存一份权重的 Shadow Copy）。
