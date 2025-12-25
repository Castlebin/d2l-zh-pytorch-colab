# 第13章 - 自然语言处理预训练

本章介绍自然语言处理中的预训练技术,从经典的词嵌入方法到现代的BERT模型。

## 📚 章节内容

本章整合了原书11个独立文件的内容,重新组织为3个综合性notebook:

### [01_词嵌入基础.ipynb](./01_词嵌入基础.ipynb)
**核心主题**: Word2Vec模型与词嵌入数据集

**涵盖内容**:
- **Word2Vec模型原理**
  - Skip-gram模型: 中心词 → 上下文词
  - CBOW模型: 上下文词 → 中心词
  - 数学推导与梯度计算
- **近似训练方法**
  - 负采样(Negative Sampling): O(|V|) → O(K)
  - 分层Softmax: O(|V|) → O(log|V|)
  - 两种方法的对比与选择
- **词嵌入数据集准备**
  - PTB数据集加载与处理
  - 词表构建与低频词处理
  - 下采样高频词
  - 中心词-上下文词提取
  - 负采样实现
  - 小批量数据加载
- **Word2Vec预训练实现**
  - 嵌入层使用
  - Skip-gram前向传播
  - 训练循环
  - 词相似度应用

**关键公式**:
- Skip-gram条件概率: $P(w_o|w_c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}$
- 负采样损失: $-\log\sigma(\mathbf{u}_o^\top \mathbf{v}_c) - \sum_{k=1}^K \log\sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_c)$
- 下采样概率: $P(\text{discard } w_i) = \max(1 - \sqrt{t/f(w_i)}, 0)$

---

### [02_GloVe与子词嵌入.ipynb](./02_GloVe与子词嵌入.ipynb)
**核心主题**: 全局向量词嵌入与子词分解技术

**涵盖内容**:
- **GloVe模型**
  - 从Word2Vec到GloVe的演进
  - 共现矩阵与全局统计
  - 共现比率的语义价值
  - 加权平方损失函数
  - GloVe vs Word2Vec对比
- **子词嵌入**
  - fastText模型: 字符n-gram方法
  - 字节对编码(BPE)算法
  - 子词提取与词表构建
  - 处理罕见词和OOV词
- **预训练词向量应用**
  - 加载GloVe预训练向量
  - 词相似度计算(K近邻)
  - 词类比任务
  - 实际应用示例

**关键公式**:
- GloVe损失: $\sum_{i,j \in \mathcal{V}} h(x_{ij})(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log x_{ij})^2$
- 权重函数: $h(x) = \begin{cases} (x/c)^\alpha & \text{if } x < c \\ 1 & \text{otherwise} \end{cases}$
- fastText词向量: $\mathbf{v}_w = \sum_{g \in \mathcal{G}_w} \mathbf{z}_g$

**技术亮点**:
- 共现比率捕捉词关系的深刻洞察
- BPE贪心算法的迭代合并过程
- WordPiece在BERT等模型中的应用

---

### [03_BERT预训练.ipynb](./03_BERT预训练.ipynb)
**核心主题**: 双向编码器表示的Transformer预训练

**涵盖内容**:
- **BERT模型架构**
  - 词表示演进历程
  - BERT输入表示设计
  - BERTEncoder实现
  - 三种嵌入的组合
- **两个预训练任务**
  - 掩码语言模型(MLM)
    - 15%掩码策略
    - 80%-10%-10%替换规则
    - MaskLM实现
  - 下一句预测(NSP)
    - IsNext vs NotNext
    - 句对关系建模
    - NextSentencePred实现
- **WikiText-2数据集**
  - 数据集选择与准备
  - NSP样本生成
  - MLM样本生成
  - 填充与批处理
- **BERT预训练实现**
  - 模型配置(Base vs Large)
  - 联合损失函数
  - 训练循环
  - 预训练BERT的使用

**关键设计**:
- **输入**: Token嵌入 + Segment嵌入 + Position嵌入
- **MLM掩码**: 80% `<mask>` + 10% random + 10% unchanged
- **总损失**: L = L_MLM + L_NSP

**BERT规格**:
| 模型 | 层数 | 隐藏单元 | 注意力头 | 参数量 |
|------|------|----------|----------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

---

## 🎯 学习目标

完成本章后,你将能够:

1. **理解词嵌入的演进**
   - 从静态词向量(Word2Vec, GloVe)到上下文相关表示(BERT)
   - 掌握不同方法的优劣与适用场景

2. **掌握核心技术**
   - 实现Word2Vec的Skip-gram和CBOW模型
   - 应用负采样和分层Softmax加速训练
   - 使用BPE进行子词分解
   - 构建和训练BERT模型

3. **处理实际数据**
   - 词表构建与数据预处理
   - 下采样与负采样策略
   - 批处理与填充技术

4. **应用预训练模型**
   - 加载和使用GloVe预训练向量
   - 词相似度和类比任务
   - BERT的微调与应用

## 📊 知识图谱

```
自然语言处理预训练
│
├─ 静态词嵌入
│  ├─ Word2Vec
│  │  ├─ Skip-gram
│  │  └─ CBOW
│  ├─ 近似训练
│  │  ├─ 负采样
│  │  └─ 分层Softmax
│  └─ GloVe
│     └─ 全局共现统计
│
├─ 子词嵌入
│  ├─ fastText (字符n-gram)
│  └─ BPE (字节对编码)
│
└─ 上下文相关表示
   └─ BERT
      ├─ Transformer编码器
      ├─ 掩码语言模型(MLM)
      └─ 下一句预测(NSP)
```

## 🔑 核心概念对比

### Word2Vec vs GloVe vs BERT

| 特性 | Word2Vec | GloVe | BERT |
|------|----------|-------|------|
| **表示类型** | 静态 | 静态 | 上下文相关 |
| **训练目标** | 预测上下文 | 拟合共现统计 | MLM + NSP |
| **模型架构** | 浅层网络 | 浅层网络 | 深层Transformer |
| **统计信息** | 局部(窗口) | 全局(共现) | 全局(自注意力) |
| **上下文** | 无 | 无 | 双向 |
| **一词多义** | ✗ | ✗ | ✓ |
| **参数量** | ~100M | ~100M | 110M-340M |
| **训练时间** | 小时 | 小时 | 天-周 |

### 近似训练方法对比

| 方法 | 计算复杂度 | 优势 | 劣势 |
|------|-----------|------|------|
| **原始Softmax** | O(\|V\|) | 精确 | 太慢 |
| **负采样** | O(K) | 简单高效 | 近似 |
| **分层Softmax** | O(log\|V\|) | 理论优雅 | 实现复杂 |

## 💡 实践技巧

### 数据预处理
```python
# 1. 构建词表
vocab = d2l.Vocab(sentences, min_freq=10)

# 2. 下采样高频词
P(discard w_i) = max(1 - sqrt(t/f(w_i)), 0)  # t=1e-4

# 3. 提取中心词-上下文对
centers, contexts = get_centers_and_contexts(corpus, max_window_size=5)

# 4. 负采样
negatives = get_negatives(contexts, vocab, K=5)
```

### BPE分词
```python
# 1. 初始化: 所有字符
symbols = ['a', 'b', ..., 'z', '_', '[UNK]']

# 2. 迭代合并最频繁的符号对
for i in range(num_merges):
    max_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_pair, token_freqs, symbols)

# 3. 贪心分词
segments = segment_BPE(tokens, symbols)
```

### BERT输入构建
```python
# 单文本
tokens = ['<cls>'] + tokens_a + ['<sep>']
segments = [0] * len(tokens)

# 文本对
tokens = ['<cls>'] + tokens_a + ['<sep>'] + tokens_b + ['<sep>']
segments = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

# 最终嵌入
embedding = token_emb + segment_emb + position_emb
```

## 🚀 性能优化

### 训练加速
1. **负采样**: 使用K=5-10代替完整Softmax
2. **子采样**: 丢弃高频词,减少训练样本
3. **批处理**: 合理设置batch_size(512-1024)
4. **多GPU**: 使用`nn.DataParallel`并行训练

### 内存优化
1. **梯度累积**: 小batch_size + 多次累积
2. **混合精度**: FP16训练节省内存
3. **动态填充**: 每批次填充到批内最大长度

## 📖 扩展阅读

### 经典论文
1. **Word2Vec**: 
   - Mikolov et al. (2013) "Efficient Estimation of Word Representations"
   - Mikolov et al. (2013) "Distributed Representations of Words and Phrases"

2. **GloVe**: 
   - Pennington et al. (2014) "GloVe: Global Vectors for Word Representation"

3. **fastText**: 
   - Bojanowski et al. (2017) "Enriching Word Vectors with Subword Information"

4. **BERT**: 
   - Devlin et al. (2018) "BERT: Pre-training of Deep Bidirectional Transformers"

### 后续发展
- **RoBERTa** (2019): 改进BERT预训练策略
- **ALBERT** (2019): 参数共享减小模型
- **ELECTRA** (2020): 判别式预训练
- **GPT-3** (2020): 扩大到175B参数

## 🎓 练习建议

### 基础练习
1. 在PTB数据集上训练Word2Vec,比较Skip-gram和CBOW
2. 实现GloVe并可视化词向量
3. 使用BPE对新语言进行分词

### 进阶练习
1. 在大规模语料上预训练BERT
2. 实现BERT在情感分类任务的微调
3. 比较不同预训练策略的效果

### 研究方向
1. 探索新的预训练任务
2. 设计更高效的子词分解算法
3. 研究多语言词嵌入

## 📈 章节统计

- **原始文件数**: 11个
- **整合后**: 3个notebook
- **代码单元**: ~60个
- **markdown单元**: ~80个
- **总行数**: ~2000行
- **预计学习时间**: 8-10小时

## 🔗 相关章节

- **前置章节**: 
  - 第10章: 注意力机制(Transformer基础)
  - 第9章: 循环神经网络(序列模型基础)

- **后续章节**: 
  - 第14章: 自然语言处理应用(BERT微调)

## ✅ 检查清单

学习本章后,确保你能够:

- [ ] 解释Word2Vec的Skip-gram和CBOW模型
- [ ] 实现负采样和分层Softmax
- [ ] 说明GloVe相对Word2Vec的改进
- [ ] 理解BPE算法的工作原理
- [ ] 构建BERT输入表示
- [ ] 解释MLM的掩码策略
- [ ] 实现完整的BERT预训练流程
- [ ] 使用预训练词向量解决实际问题

---

## 🎉 小结

本章系统介绍了NLP预训练技术的发展历程:

1. **静态词嵌入**: Word2Vec和GloVe奠定了词向量的基础
2. **子词嵌入**: fastText和BPE解决了OOV和形态学问题  
3. **上下文表示**: BERT开启了预训练-微调的新范式

这些技术构成了现代NLP的基石,掌握它们对理解后续的GPT、T5等模型至关重要!

**下一章预告**: 我们将学习如何将预训练的BERT模型应用到具体的NLP任务中,包括情感分析、自然语言推理等。

---

*整理完成时间: 2025年10月27日*  
*源文件: chapter_natural-language-processing-pretraining (11个文件)*  
*目标文件: 3个综合notebook + README*
