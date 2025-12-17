# 第8章: 循环神经网络 (RNN)

> 处理序列数据: 从RNN到LSTM/GRU

## 📚 本章内容

本章介绍专门处理序列数据的神经网络架构。

### Notebook列表

1. **01_RNN基础与文本序列.ipynb** ⭐⭐⭐
   - 序列模型动机
   - 文本预处理(分词、词表、数值化)
   - RNN原理与实现
   - 字符级语言模型

2. **02_LSTM与GRU.ipynb** ⭐⭐⭐
   - 梯度消失/爆炸问题
   - LSTM: 门控机制与长期记忆
   - GRU: 简化的门控单元
   - PyTorch实现

3. **03_序列模型应用.ipynb** ⭐⭐
   - 双向RNN
   - 深度RNN
   - Seq2Seq与编码器-解码器
   - 机器翻译示例

4. **04_束搜索与序列生成.ipynb** ⭐⭐⭐
   - 序列解码问题
   - 贪心搜索与局限性
   - 穷举搜索的不可行性
   - 束搜索算法详解
   - 长度归一化与参数调优

## 🎯 学习目标

- ✅ 理解为什么需要RNN处理序列
- ✅ 掌握RNN的隐状态机制
- ✅ 理解LSTM/GRU如何解决梯度问题
- ✅ 实现字符级/词级语言模型
- ✅ 了解Seq2Seq架构
- ✅ 掌握束搜索等序列解码策略

## 📖 核心概念

### 1. 为什么需要RNN?

**序列数据无处不在**:
- 文本: \"狗咬人\" ≠ \"人咬狗\"
- 时间序列: 股价、天气
- 语音: 声音波形
- 视频: 帧序列

**CNN/MLP的局限**:
- ❌ 输入长度固定
- ❌ 无法共享参数
- ❌ 没有\"记忆\"

### 2. RNN原理

**核心公式**:
$$\\mathbf{H}_t = \\phi(\\mathbf{X}_t \\mathbf{W}_{xh} + \\mathbf{H}_{t-1} \\mathbf{W}_{hh} + \\mathbf{b}_h)$$

**关键特点**:
- 隐状态 $\\mathbf{H}_t$: \"记忆\"
- 参数共享: 所有时间步用同样的 $\\mathbf{W}$
- 循环结构: $\\mathbf{H}_{t-1} \\to \\mathbf{H}_t$

### 3. 梯度问题

**梯度消失**:
$$\\frac{\\partial L}{\\partial \\mathbf{W}_{hh}} \\propto \\prod_{t=1}^T \\mathbf{W}_{hh}^T$$

如果 $\\|\\mathbf{W}_{hh}\\| < 1$, 梯度指数衰减!

**梯度爆炸**:
如果 $\\|\\mathbf{W}_{hh}\\| > 1$, 梯度指数增长!

### 4. LSTM解决方案

**三个门**:
- **遗忘门** $\\mathbf{F}_t$: 忘记什么
- **输入门** $\\mathbf{I}_t$: 记住什么
- **输出门** $\\mathbf{O}_t$: 输出什么

**记忆元**:
$$\\mathbf{C}_t = \\mathbf{F}_t \\odot \\mathbf{C}_{t-1} + \\mathbf{I}_t \\odot \\tilde{\\mathbf{C}}_t$$

**优势**: 梯度高速公路,缓解消失问题

### 5. GRU简化版

只有两个门:
- **重置门** $\\mathbf{R}_t$
- **更新门** $\\mathbf{Z}_t$

更少参数,训练更快!

## 🔑 关键公式对比

| 模型 | 隐状态更新 | 门数 | 参数量 |
|------|-----------|------|--------|
| **RNN** | $\\mathbf{H}_t = \\tanh(...)$ | 0 | 少 |
| **LSTM** | $\\mathbf{H}_t = \\mathbf{O}_t \\odot \\tanh(\\mathbf{C}_t)$ | 3 | 多 |
| **GRU** | $\\mathbf{H}_t = (1-\\mathbf{Z}_t) \\odot \\mathbf{H}_{t-1} + \\mathbf{Z}_t \\odot \\tilde{\\mathbf{H}}_t$ | 2 | 中 |

## 💡 常见问题

### Q1: RNN vs CNN?
| 方面 | RNN | CNN |
|------|-----|-----|
| 输入 | 序列(变长) | 图像(固定) |
| 结构 | 循环 | 层叠 |
| 参数 | 共享(时间) | 共享(空间) |
| 并行 | 困难 | 容易 |
| 用途 | 文本/时序 | 图像/视频 |

### Q2: 如何选择RNN/LSTM/GRU?
- **短序列**: RNN即可
- **长序列**: LSTM/GRU
- **资源受限**: GRU (参数少)
- **需要长期记忆**: LSTM

### Q3: 梯度裁剪如何工作?
```python
# 限制梯度范数
if grad_norm > threshold:
    grad *= threshold / grad_norm
```

防止梯度爆炸!

### Q4: 困惑度(Perplexity)是什么?
$$\\text{PPL} = \\exp(\\text{CrossEntropy})$$

越低越好! PPL=100表示模型平均在100个词中\"困惑\"。

### Q5: 束搜索中的束宽如何选择?
| 束宽k | 速度 | 质量 | 适用场景 |
|-------|------|------|----------|
| k=1 | 最快 | 低 | 贪心搜索 |
| k=5 | 快 | 中 | 实时应用 |
| k=10 | 中 | 较高 | 标准设置(推荐) |
| k=50 | 慢 | 高 | 离线翻译 |

### Q6: 什么是长度归一化?
$$\\text{score} = \\frac{1}{L^\\alpha} \\sum_{t=1}^L \\log P(y_t)$$

防止模型偏向生成短序列! 典型值: $\\alpha = 0.75$

## 📊 应用场景

### 语言模型
```python
输入: \"深度学习很\"
输出: \"有趣\" (概率最高)
```

### 机器翻译
```python
输入: \"I love you\"
输出: \"我爱你\"
```

### 情感分析
```python
输入: \"这部电影太棒了!\"
输出: \"正面\" (0.95)
```

### 时间序列预测
```python
输入: [股价_{t-10}, ..., 股价_{t-1}]
输出: 股价_t
```

## 🛠️ 实践建议

### 训练技巧

1. **梯度裁剪** (必须!)
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **学习率调度**
   ```python
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
   ```

3. **Dropout正则化**
   ```python
   nn.LSTM(input_size, hidden_size, dropout=0.5)
   ```

4. **批处理技巧**
   - 序列填充(padding)
   - 打包序列(pack_padded_sequence)

5. **束搜索解码**
   ```python
   # 生成序列时使用束搜索替代贪心
   output = beam_search(model, input, beam_size=10, alpha=0.75)
   ```

### 调试技巧

1. **检查梯度**
   ```python
   for name, param in model.named_parameters():
       print(f\"{name}: {param.grad.norm()}\")
   ```

2. **可视化隐状态**
   ```python
   plt.imshow(hidden_states.detach().cpu().numpy())
   ```

3. **从简单开始**
   - 先用小数据集
   - 单层RNN
   - 逐步增加复杂度

## 📚 推荐阅读

### 必读论文

1. **LSTM** (1997) ⭐⭐⭐
   - *Long Short-Term Memory*
   - Hochreiter & Schmidhuber
   - [PDF](https://www.bioinf.jku.at/publications/older/2604.pdf)

2. **GRU** (2014)
   - *Learning Phrase Representations using RNN Encoder-Decoder*
   - Cho et al.
   - [PDF](https://arxiv.org/abs/1406.1078)

3. **Seq2Seq** (2014)
   - *Sequence to Sequence Learning with Neural Networks*
   - Sutskever, Vinyals, Le
   - [PDF](https://arxiv.org/abs/1409.3215)

4. **Beam Search优化** (2016)
   - *Google's Neural Machine Translation System*
   - Wu et al.
   - [PDF](https://arxiv.org/abs/1609.08144)

## 🎓 学习路径

### 第1周: RNN基础
- [ ] 理解序列模型动机
- [ ] 掌握文本预处理
- [ ] 实现简单RNN
- [ ] 训练字符级语言模型

### 第2周: LSTM/GRU
- [ ] 理解梯度问题
- [ ] 掌握门控机制
- [ ] 对比LSTM/GRU性能
- [ ] 实现情感分析

### 第3周: 序列应用
- [ ] 学习Seq2Seq
- [ ] 实现机器翻译
- [ ] 掌握束搜索算法
- [ ] 对比不同解码策略
- [ ] 了解注意力机制
- [ ] 项目: 对话系统

## 🔧 PyTorch实现

### 简洁实现RNN
```python
# 单层RNN
rnn = nn.RNN(input_size=vocab_size, 
             hidden_size=256, 
             num_layers=1)

# 多层LSTM with Dropout
lstm = nn.LSTM(input_size=vocab_size,
               hidden_size=256,
               num_layers=2,
               dropout=0.5)

# GRU
gru = nn.GRU(input_size=vocab_size,
             hidden_size=256,
             num_layers=2)
```

### 完整模型
```python
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout=0.5)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, state):
        x = self.embedding(x)  # (seq_len, batch, embed)
        y, state = self.rnn(x, state)  # (seq_len, batch, hidden)
        output = self.fc(y)  # (seq_len, batch, vocab)
        return output, state
```

## 🎉 总结

### 核心收获

1. **RNN**: 处理序列数据的基础架构
2. **隐状态**: RNN的\"记忆机制\"
3. **LSTM/GRU**: 解决梯度消失,学习长期依赖
4. **应用**: 语言模型、翻译、情感分析等

### RNN的局限

- ❌ 顺序计算,无法并行
- ❌ 长序列效率低
- ❌ 远距离依赖仍困难

### 未来方向

- **Transformer** (第9章): 自注意力机制
- **BERT/GPT**: 预训练语言模型
- **高效RNN**: SRU, QRNN

---

**下一章**: 注意力机制与Transformer 🚀