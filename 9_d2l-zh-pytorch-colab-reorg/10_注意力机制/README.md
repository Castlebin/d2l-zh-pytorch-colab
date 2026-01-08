# Chapter 9: 注意力机制

本章介绍深度学习中最重要的创新之一——注意力机制,从基础概念到Transformer架构的完整实现。

## 📚 内容概览

### Notebook 1: 注意力机制基础
**文件**: `01_注意力机制基础.ipynb`

**核心内容**:
- 注意力机制的生物学动机
- 查询(Query)、键(Key)、值(Value)三元组
- 注意力评分函数(加性注意力 vs 点积注意力)
- Nadaraya-Watson核回归作为注意力示例

**关键公式**:
```
注意力汇聚: f(q, K, V) = Σ α(q, k_i) v_i
注意力权重: α(q, k_i) = softmax(a(q, k_i))

加性注意力: a(q, k) = w_v^T tanh(W_q q + W_k k)
点积注意力: a(q, k) = (q^T k) / √d
```

**实践项目**:
- 从零实现加性注意力和点积注意力
- Nadaraya-Watson核回归的训练与可视化
- 注意力权重热力图分析

---

### Notebook 2: Seq2Seq注意力与多头注意力
**文件**: `02_Seq2Seq注意力与多头注意力.ipynb`

**核心内容**:
- Seq2seq模型的信息瓶颈问题
- Bahdanau注意力机制
- 多头注意力(Multi-Head Attention)
- 注意力权重可视化与对齐分析

**关键公式**:
```
Bahdanau注意力:
  c_t' = Σ α(s_{t'-1}, h_t) h_t
  α = softmax(v^T tanh(W_s s + W_h h))

多头注意力:
  head_i = Attention(W_i^Q q, W_i^K k, W_i^V v)
  MultiHead(q,k,v) = W_o [head_1; ...; head_h]
```

**实践项目**:
- 带Bahdanau注意力的seq2seq模型
- 序列反转任务(可视化注意力对齐)
- 多头注意力的实现与不同头的模式分析

**性能对比**:
| 模型 | 评分函数 | 复杂度 | 用途 |
|------|----------|--------|------|
| 加性注意力 | tanh(W_q q + W_k k) | O(d²) | Seq2seq |
| 点积注意力 | q^T k / √d | O(d) | Transformer |
| 多头注意力 | h个点积头 | O(hd) | Transformer |

---

### Notebook 3: 自注意力与Transformer
**文件**: `03_自注意力与Transformer.ipynb`

**核心内容**:
- 自注意力机制(Self-Attention)
- 位置编码(Positional Encoding)
- CNN vs RNN vs Self-Attention对比
- 完整Transformer架构实现

**关键公式**:
```
自注意力:
  SelfAttention(X) = softmax(QK^T / √d) V
  其中 Q = XW^Q, K = XW^K, V = XW^V

正弦位置编码:
  PE(i, 2j)   = sin(i / 10000^(2j/d))
  PE(i, 2j+1) = cos(i / 10000^(2j/d))
```

**Transformer组件**:
1. **编码器块**: 多头自注意力 → 残差&归一化 → FFN → 残差&归一化
2. **解码器块**: 掩码自注意力 → 交叉注意力 → FFN (均带残差和归一化)
3. **位置前馈网络**: FFN(x) = ReLU(xW1 + b1)W2 + b2
4. **掩码机制**: 因果掩码(防止看到未来) + Padding掩码(忽略填充)

**实践项目**:
- 从零实现完整Transformer (Base版本)
- 位置编码的可视化与相对位置验证
- 不同掩码类型的作用分析

---

## 🎯 学习路径

### 初学者路径
1. **第1周**: Notebook 1 - 理解QKV概念,实现简单注意力
2. **第2周**: Notebook 2 - 学习Bahdanau注意力和多头机制
3. **第3周**: Notebook 3 - 掌握自注意力和位置编码
4. **第4周**: 实现完整Transformer并在简单任务上训练

### 进阶路径
1. 在真实翻译数据集(如WMT)上训练Transformer
2. 实现Transformer的变种(如Pre-LN, Flash Attention)
3. 探索长序列优化(Linformer, Performer)
4. 应用于CV领域(Vision Transformer)

---

## 📊 架构对比

### CNN vs RNN vs Self-Attention

| 特性        | CNN     | RNN    | Self-Attention |
|-----------|---------|--------|----------------|
| **计算复杂度** | O(knd²) | O(nd²) | O(n²d)         |
| **顺序操作**  | O(1)    | O(n)   | O(1)           |
| **最大路径**  | O(n/k)  | O(n)   | O(1) ✅         |
| **并行计算**  | ✅ 高     | ❌ 低    | ✅ 高            |
| **长距离依赖** | ❌ 需多层   | ❌ 梯度问题 | ✅ 直接连接         |
| **位置信息**  | ✅ 隐式    | ✅ 天然   | ❌ 需编码          |
| **内存消耗**  | 低       | 低      | 高(O(n²))       |

**结论**: 
- 短序列 → Self-Attention (最佳)
- 长序列 → RNN或稀疏注意力
- 局部特征 → CNN

---

## 💡 关键概念速查

### 1. 注意力权重计算
```python
# 评分 → 归一化 → 加权求和
scores = score_function(query, keys)      # 任意评分函数
weights = softmax(scores)                 # 归一化为概率
output = weighted_sum(weights, values)    # 加权平均
```

### 2. 多头注意力优势
- **多子空间**: 每个头关注不同模式
- **更强表达**: 捕获多种依赖关系
- **参数效率**: 相比单头仅增加1倍参数

### 3. 位置编码设计
- **频率递减**: 低维高频(局部) + 高维低频(全局)
- **相对位置**: 支持线性变换表示偏移
- **外推性**: 可处理更长序列

### 4. Transformer训练技巧
- **学习率预热**: 前N步线性增加学习率
- **标签平滑**: 防止过拟合
- **Dropout**: 在注意力权重和FFN中应用
- **梯度裁剪**: 防止梯度爆炸

---

## 🔧 实现细节

### 掩码类型

```python
# 1. Padding掩码(忽略填充)
def create_padding_mask(seq, pad_id=0):
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)

# 2. 因果掩码(防止看到未来)
def create_causal_mask(size):
    return torch.tril(torch.ones(size, size))

# 3. 组合掩码
combined_mask = padding_mask & causal_mask
```

### 缩放因子的重要性

```python
# 为什么要除以√d?
# 点积方差随维度线性增长: Var(q^T k) = d
# 大值 → softmax饱和 → 梯度消失
# 除以√d使方差稳定在1

scores = Q @ K.T / math.sqrt(d_k)
```

### 残差连接与层归一化

```python
# Post-LN (原始Transformer)
def encoder_block(x):
    x = LayerNorm(x + MultiHeadAttention(x))
    x = LayerNorm(x + FFN(x))
    return x

# Pre-LN (更稳定,深层网络更好)
def encoder_block_pre_ln(x):
    x = x + MultiHeadAttention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
    return x
```

---

## 📈 性能优化

### 1. 计算效率
- **Flash Attention**: 优化GPU内存访问模式
- **稀疏注意力**: 只计算部分注意力权重
- **线性注意力**: 将复杂度从O(n²)降至O(n)

### 2. 内存优化
- **梯度检查点**: 牺牲计算换内存
- **混合精度**: 使用FP16训练
- **序列切分**: 长序列分块处理

### 3. 长序列处理
- **Linformer**: 低秩近似,复杂度O(n)
- **Performer**: 随机特征近似注意力
- **Longformer**: 滑动窗口 + 全局注意力

---

## 🎓 理论深入

### 注意力机制的本质

**从优化角度**:
- 注意力是一种软性的键值查询
- 等价于核密度估计的加权平均
- 可视为记忆网络的可微版本

**从信息论角度**:
- 注意力权重是条件概率分布
- 最大化输入与输出的互信息
- 选择性提取相关信息,过滤噪声

**从神经科学角度**:
- 模拟人类的选择性注意
- 非自主性提示(键) + 自主性提示(查询)
- 稀缺资源的高效分配

### Transformer为什么有效?

1. **归纳偏置少**: 不假设数据结构,数据驱动学习
2. **全局感受野**: 直接建模长距离依赖
3. **并行训练**: 利用现代GPU并行能力
4. **可扩展性**: 更大模型 + 更多数据 = 更好性能

---

## 🚀 应用场景

### NLP领域
- **预训练模型**: BERT, GPT, T5, BART
- **机器翻译**: Google Translate, DeepL
- **文本生成**: ChatGPT, Claude
- **问答系统**: SQuAD, Natural Questions

### CV领域
- **图像分类**: Vision Transformer (ViT)
- **目标检测**: DETR (Detection Transformer)
- **图像生成**: DALL-E, Stable Diffusion

### 多模态
- **图文匹配**: CLIP, ALIGN
- **视频理解**: TimeSformer, VideoMAE
- **语音识别**: Conformer, Whisper

---

## 📖 推荐阅读

### 经典论文
1. **Attention机制**:
   - Bahdanau et al. (2015) - "Neural Machine Translation by Jointly Learning to Align and Translate"
   
2. **Transformer**:
   - Vaswani et al. (2017) - "Attention Is All You Need" ⭐

3. **预训练模型**:
   - Devlin et al. (2019) - "BERT: Pre-training of Deep Bidirectional Transformers"
   - Radford et al. (2019) - "Language Models are Unsupervised Multitask Learners" (GPT-2)

4. **长序列优化**:
   - Wang et al. (2020) - "Linformer: Self-Attention with Linear Complexity"
   - Choromanski et al. (2021) - "Rethinking Attention with Performers"

### 博客资源
- The Illustrated Transformer (Jay Alammar)
- The Annotated Transformer (Harvard NLP)
- Attention? Attention! (Lilian Weng)

---

## 🛠️ 调试技巧

### 常见问题

1. **注意力权重全是NaN**
   - 检查: 除以√d是否遗漏
   - 检查: 掩码是否设置为-inf而非0
   - 检查: 梯度是否爆炸(添加梯度裁剪)

2. **训练不收敛**
   - 学习率预热(前4000步线性增加)
   - 标签平滑(smoothing=0.1)
   - 增加Dropout

3. **内存溢出**
   - 减小batch_size
   - 使用梯度累积
   - 启用混合精度训练

4. **推理速度慢**
   - 使用缓存机制(保存已计算的K, V)
   - Beam search时共享编码器输出
   - 量化模型(INT8)

---

## 🎯 练习题

### 基础题
1. 手动计算一个3×3矩阵的自注意力输出
2. 证明位置编码的相对位置性质
3. 分析多头注意力的参数量

### 进阶题
1. 实现相对位置编码(如T5, DeBERTa)
2. 比较Pre-LN和Post-LN的训练曲线
3. 实现Flash Attention的简化版本

### 项目题
1. 在IWSLT数据集上训练英德翻译模型
2. 可视化不同层不同头的注意力模式
3. 实现Beam Search解码并优化速度

---

## 📦 依赖环境

```bash
# 基础环境
pip install torch torchvision
pip install numpy matplotlib seaborn

# 进阶工具
pip install transformers  # Hugging Face
pip install tensorboard   # 可视化
pip install sacrebleu     # 翻译评估
```

---

## 🌟 本章亮点

1. **从零实现**: 所有代码不依赖d2l库,纯PyTorch实现
2. **可视化丰富**: 注意力权重、位置编码、掩码机制全面可视化
3. **理论与实践**: 深入剖析数学原理,配合完整代码实现
4. **前沿扩展**: 介绍Flash Attention等最新优化技术

---

**下一章预告**: Chapter 10 - 优化算法 (SGD, Adam, 学习率调度等)

Happy Learning! 🚀

