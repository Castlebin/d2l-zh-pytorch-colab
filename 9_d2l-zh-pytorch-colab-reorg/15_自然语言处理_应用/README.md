# 第14章:自然语言处理应用

本章介绍自然语言处理(NLP)的两个重要应用领域:情感分析和自然语言推理,以及如何使用BERT进行模型微调。

## 📚 章节内容

### [01_情感分析.ipynb](01_情感分析.ipynb)
**主题**: 使用RNN和CNN进行情感分析

**核心内容**:
- 情感分析任务介绍与应用场景
- IMDb电影评论数据集的加载与预处理
- 基于双向LSTM的情感分类模型
- 基于textCNN的情感分类模型
- RNN vs CNN模型对比分析

**关键概念**:
- 情感分析(Sentiment Analysis)
- 文本分类(Text Classification)
- 双向循环神经网络(Bidirectional RNN)
- 一维卷积神经网络(1D CNN)
- 最大时间汇聚(Max-over-time Pooling)
- 预训练词向量(GloVe)

**代码实现**:
```python
# 双向LSTM模型
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers):
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, 
                              num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

# textCNN模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels):
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv1d(2 * embed_size, c, k) 
            for c, k in zip(num_channels, kernel_sizes)])
        self.decoder = nn.Linear(sum(num_channels), 2)
```

**数据集**: IMDb电影评论(25,000训练 + 25,000测试)

---

### [02_自然语言推理.ipynb](02_自然语言推理.ipynb)
**主题**: 基于注意力机制的自然语言推理

**核心内容**:
- 自然语言推理(NLI)任务定义
- SNLI数据集的结构与处理
- 可分解注意力模型的三个步骤
  - 注意(Attending): 软对齐
  - 比较(Comparing): 特征提取
  - 聚合(Aggregating): 分类决策
- 模型训练与评估

**关键概念**:
- 自然语言推理(Natural Language Inference)
- 前提(Premise)与假设(Hypothesis)
- 蕴涵(Entailment)/矛盾(Contradiction)/中性(Neutral)
- 可分解注意力(Decomposable Attention)
- 软对齐(Soft Alignment)
- 分解技巧(Decomposition Trick)

**模型架构**:
```
输入: (前提, 假设)
  ↓
注意步骤: 计算注意力权重, 软对齐
  e_ij = f(a_i)^T f(b_j)
  β_i = Σ softmax(e_ij) * b_j
  α_j = Σ softmax(e_ij) * a_i
  ↓
比较步骤: 比较对齐的词元
  v_A,i = g([a_i, β_i])
  v_B,j = g([b_j, α_j])
  ↓
聚合步骤: 求和并分类
  v_A = Σ v_A,i
  v_B = Σ v_B,j
  y_hat = h([v_A, v_B])
  ↓
输出: 蕴涵/矛盾/中性
```

**数据集**: SNLI(550,000训练 + 10,000测试)

**复杂度分析**:
- 传统方法: O(m × n) - 二次复杂度
- 分解技巧: O(m + n) - 线性复杂度

---

### [03_BERT微调.ipynb](03_BERT微调.ipynb)
**主题**: BERT在序列级和词元级任务中的微调

**核心内容**:
- BERT微调的基本原理
- 序列级应用
  - 单文本分类(情感分析、语言可接受性)
  - 文本对分类(NLI、语义相似度)
- 词元级应用
  - 文本标注(词性标注、NER)
  - 问答(SQuAD)
- BERT用于NLI的完整实现

**关键概念**:
- 预训练与微调(Pre-training & Fine-tuning)
- 迁移学习(Transfer Learning)
- 最小架构更改(Minimal Architecture Changes)
- &lt;CLS&gt; token表示
- Segment Embeddings
- 陈旧梯度(Stale Gradients)

**微调策略**:
| 参数类型 | 更新策略 |
|---------|---------|
| 额外层参数(输出层) | 从零开始学习 |
| BERT编码器参数 | 微调(Fine-tune) |
| BERT隐藏层参数 | 微调(Fine-tune) |
| MLM/NSP相关参数 | 不更新(Stale) |

**任务对比**:
```
序列级任务:
┌─────────────────────┬──────────────────────┐
│ 单文本分类           │ 文本对分类/回归        │
├─────────────────────┼──────────────────────┤
│ [CLS] 文本 [SEP]     │ [CLS] 文本1 [SEP]     │
│        ↓            │       文本2 [SEP]     │
│   CLS表示 → MLP     │        ↓              │
│        ↓            │   CLS表示 → MLP       │
│     类别分布         │        ↓              │
│                     │  类别分布/连续值       │
└─────────────────────┴──────────────────────┘

词元级任务:
┌─────────────────────┬──────────────────────┐
│ 文本标注             │ 问答                  │
├─────────────────────┼──────────────────────┤
│ [CLS] w1 w2 ... [SEP]│ [CLS] 问题 [SEP]      │
│   ↓   ↓  ↓          │       段落 [SEP]      │
│ 每个词元表示 → FC    │        ↓              │
│   ↓   ↓  ↓          │  开始FC → 分数s_i     │
│  标签1 标签2 ...     │  结束FC → 分数e_j     │
│                     │        ↓              │
│                     │ argmax(s_i + e_j)     │
└─────────────────────┴──────────────────────┘
```

---

## 🔄 章节知识图谱

```
自然语言处理应用
├── 情感分析(文本分类)
│   ├── 数据处理
│   │   ├── IMDb数据集
│   │   ├── 词元化
│   │   ├── 截断与填充
│   │   └── 词表构建
│   ├── RNN方法
│   │   ├── 双向LSTM
│   │   ├── 连结首尾隐状态
│   │   ├── 预训练GloVe
│   │   └── 全连接分类器
│   └── CNN方法
│       ├── 一维卷积
│       ├── 多尺度卷积核
│       ├── 最大时间汇聚
│       └── 全连接分类器
├── 自然语言推理(文本对分类)
│   ├── 数据处理
│   │   ├── SNLI数据集
│   │   ├── 前提-假设对
│   │   └── 三类标签
│   └── 可分解注意力
│       ├── 注意步骤(软对齐)
│       ├── 比较步骤(特征提取)
│       ├── 聚合步骤(分类)
│       └── 分解技巧(线性复杂度)
└── BERT微调(迁移学习)
    ├── 序列级任务
    │   ├── 单文本分类
    │   │   ├── 情感分析
    │   │   └── 语言可接受性
    │   └── 文本对分类/回归
    │       ├── 自然语言推理
    │       └── 语义相似度
    └── 词元级任务
        ├── 文本标注
        │   ├── 词性标注
        │   └── 命名实体识别
        └── 问答
            ├── 阅读理解
            ├── 开始位置预测
            └── 结束位置预测
```

---

## 📊 模型对比

### 情感分析模型对比

| 模型          | 架构             | 优点                               | 缺点                             | 适用场景                        |
|-------------|----------------|----------------------------------|--------------------------------|-----------------------------|
| **BiLSTM**  | 双向循环神经网络       | • 捕获长距离依赖<br>• 保留序列顺序<br>• 理解上下文 | • 训练慢(序列处理)<br>• 梯度问题<br>• 参数多 | • 需要序列信息<br>• 长距离依赖重要       |
| **textCNN** | 一维卷积+池化        | • 并行计算快<br>• 多尺度特征<br>• 参数少      | • 难捕获长依赖<br>• 丢失顺序信息           | • 文本分类<br>• 快速推理<br>• 关键词重要 |
| **BERT微调**  | Transformer编码器 | • SOTA效果<br>• 预训练知识<br>• 通用性强    | • 计算资源大<br>• 推理慢               | • 高精度需求<br>• 资源充足           |

### NLI模型对比

| 模型         | 复杂度       | 参数量 | 特点          |
|------------|-----------|-----|-------------|
| **可分解注意力** | O(m+n)    | 小   | 简单高效,可解释性强  |
| **ESIM**   | O(m×n)    | 中   | 序列建模,效果好    |
| **BERT微调** | O((m+n)²) | 大   | SOTA效果,计算量大 |

---

## 💡 核心技术要点

### 1. 预训练词向量

**GloVe使用**:
```python
# 加载预训练GloVe
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]

# 初始化embedding层
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False  # 固定词向量
```

**优势**:
- 减少过拟合
- 利用大规模语料库知识
- 加速收敛

### 2. 注意力机制

**可分解注意力核心**:
```python
# 计算注意力权重
e = torch.bmm(f(A), f(B).permute(0, 2, 1))  # (batch, m, n)

# 软对齐
beta = torch.bmm(F.softmax(e, dim=-1), B)  # 假设→前提
alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)  # 前提→假设
```

**分解技巧**:
- 不是$f([a_i, b_j])$: O(m×n)次计算
- 而是$f(a_i)^T f(b_j)$: O(m+n)次计算
- 大幅降低计算复杂度

### 3. BERT微调技巧

**参数更新策略**:
```python
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        self.encoder = bert.encoder  # 微调
        self.hidden = bert.hidden    # 微调
        self.output = nn.Linear(256, 3)  # 从零开始
```

**训练设置**:
- 小学习率(1e-4)防止破坏预训练权重
- warmup学习率调度
- 梯度裁剪
- 允许陈旧梯度(`ignore_stale_grad=True`)

---

## 🔬 实验技巧

### 数据预处理

1. **序列长度选择**:
   - IMDb: 500 (覆盖大部分评论)
   - SNLI: 50 (句子较短)
   - BERT: 128-512 (根据资源调整)

2. **词表构建**:
   ```python
   vocab = d2l.Vocab(tokens, min_freq=5, reserved_tokens=['<pad>'])
   ```
   - `min_freq=5`: 过滤低频词
   - 保留特殊token

### 训练优化

1. **批量大小**:
   - RNN/CNN: 64
   - 注意力: 256
   - BERT: 512 (显存允许的话)

2. **学习率**:
   - 从零训练: 1e-3
   - 微调预训练模型: 1e-4

3. **正则化**:
   - Dropout(0.2-0.5)
   - 预训练词向量固定
   - 早停(Early Stopping)

---

## 📈 性能基准

### IMDb情感分析

| 模型        | 测试准确率 | 训练时间 | 推理速度 |
|-----------|-------|------|------|
| BiLSTM    | ~85%  | 慢    | 中    |
| textCNN   | ~85%  | 快    | 快    |
| BERT-base | ~93%  | 很慢   | 慢    |

### SNLI自然语言推理

| 模型        | 测试准确率 | 参数量  |
|-----------|-------|------|
| 可分解注意力    | ~86%  | 300K |
| ESIM      | ~88%  | 4.3M |
| BERT-base | ~91%  | 110M |

---

## 🎯 应用场景

### 情感分析

**商业应用**:
- 产品评论分析
- 社交媒体监控
- 客户反馈分析
- 品牌声誉管理

**示例**:
```python
# 预测电影评论情感
predict_sentiment(net, vocab, 'this movie is so great')
# 输出: 'positive'
```

### 自然语言推理

**应用领域**:
- 信息检索(查询-文档匹配)
- 问答系统(答案验证)
- 文本摘要(冗余消除)
- 对话系统(逻辑一致性检查)

**示例**:
```python
# 判断逻辑关系
premise = ['he', 'is', 'good']
hypothesis = ['he', 'is', 'bad']
predict_snli(net, vocab, premise, hypothesis)
# 输出: 'contradiction'
```

### BERT微调

**通用NLP任务**:
- 序列级: 分类、回归
- 词元级: 标注、抽取
- 生成级: 摘要、翻译(with decoder)

---

## 🔧 代码模板

### 情感分析完整流程

```python
# 1. 加载数据
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size=64, num_steps=500)

# 2. 定义模型
net = BiRNN(len(vocab), embed_size=100, num_hiddens=100, num_layers=2)

# 3. 加载预训练词向量
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
net.embedding.weight.data.copy_(glove_embedding[vocab.idx_to_token])
net.embedding.weight.requires_grad = False

# 4. 训练
trainer = torch.optim.Adam(net.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs=5, devices)

# 5. 预测
predict_sentiment(net, vocab, 'this movie is so great')
```

### NLI完整流程

```python
# 1. 加载数据
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size=256, num_steps=50)

# 2. 定义模型
net = DecomposableAttention(vocab, embed_size=100, num_hiddens=200)

# 3. 加载预训练词向量
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
net.embedding.weight.data.copy_(glove_embedding[vocab.idx_to_token])

# 4. 训练
trainer = torch.optim.Adam(net.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs=4, devices)

# 5. 预测
predict_snli(net, vocab, ['he', 'is', 'good'], ['he', 'is', 'bad'])
```

### BERT微调流程

```python
# 1. 加载预训练BERT
bert, vocab = load_pretrained_model('bert.small', ...)

# 2. 定义分类器
net = BERTClassifier(bert)

# 3. 准备数据
train_set = SNLIBERTDataset(train_data, max_len=128, vocab=vocab)
train_iter = DataLoader(train_set, batch_size=512, shuffle=True)

# 4. 微调
trainer = torch.optim.Adam(net.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs=5, devices)
```

---

## 📚 扩展阅读

### 论文推荐

1. **情感分析**:
   - Kim, 2014: "Convolutional Neural Networks for Sentence Classification" (textCNN)
   - Maas et al., 2011: "Learning Word Vectors for Sentiment Analysis" (IMDb数据集)

2. **自然语言推理**:
   - Bowman et al., 2015: "A large annotated corpus for learning natural language inference" (SNLI)
   - Parikh et al., 2016: "A Decomposable Attention Model for Natural Language Inference"

3. **BERT**:
   - Devlin et al., 2019: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
   - Liu et al., 2019: "RoBERTa: A Robustly Optimized BERT Pretraining Approach"

### 相关资源

- [SNLI数据集](https://nlp.stanford.edu/projects/snli/)
- [IMDb数据集](http://ai.stanford.edu/~amaas/data/sentiment/)
- [GloVe词向量](https://nlp.stanford.edu/projects/glove/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---

## ✅ 学习检查清单

完成本章后,你应该能够:

- [ ] 理解情感分析任务,实现RNN和CNN模型
- [ ] 加载和预处理IMDb、SNLI等文本数据集
- [ ] 实现可分解注意力模型,理解分解技巧
- [ ] 解释自然语言推理的三种关系类型
- [ ] 使用预训练词向量初始化embedding层
- [ ] 理解BERT微调的基本原理和参数更新策略
- [ ] 区分序列级和词元级NLP任务
- [ ] 实现BERT在NLI任务上的微调
- [ ] 对比不同模型的优缺点和适用场景
- [ ] 将学到的技术应用到实际NLP项目中

---

## 🎓 练习建议

### 基础练习

1. 修改超参数(学习率、批量大小、隐藏单元数)观察效果
2. 尝试不同的预训练词向量(GloVe 50d, 200d, 300d)
3. 实现Amazon评论数据集的加载和训练
4. 可视化注意力权重矩阵

### 进阶练习

5. 实现ESIM模型并与可分解注意力对比
6. 尝试ensemble多个模型提高准确率
7. 实现BERT-base微调,对比bert.small
8. 添加位置编码到textCNN模型
9. 设计混合模型(CNN+RNN或CNN+Attention)

### 项目练习

10. 构建多语言情感分析系统
11. 开发基于NLI的问答验证系统
12. 实现文本相似度计算API
13. 构建新闻文章搜索引擎(使用BERT)
14. 开发实时社交媒体情感监控dashboard

---

**上一章**: [13_自然语言处理预训练](../13_自然语言处理预训练/README.md)

**下一章**: [15_推荐系统](../15_推荐系统/README.md) (如果有)

---

*最后更新: 2024*
