# 03 线性神经网络

本章介绍最基础的神经网络模型 - **线性模型**,包括回归和分类两大类问题。

## 📚 学习内容

### 1. 线性回归 (01_线性回归.ipynb)
- **回归问题**: 预测连续值
- **核心概念**:
  - 线性模型: $\hat{y} = \mathbf{w}^T \mathbf{x} + b$
  - 均方误差损失: $L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$
  - 梯度下降优化
- **实现方式**:
  - 从零开始实现(手动实现数据迭代、模型、损失、SGD)
  - PyTorch简洁实现(使用`nn.Linear`, `nn.MSELoss`, `torch.optim.SGD`)

### 2. Softmax回归与图像分类 (02_Softmax回归与图像分类.ipynb)
- **分类问题**: 预测离散类别
- **Fashion-MNIST数据集**: 10类服饰图像识别
- **核心概念**:
  - Softmax函数: 将实数转为概率分布
  - 交叉熵损失: 衡量预测分布与真实分布的差异
  - 独热编码: 类别的向量表示
- **实现方式**:
  - 从零开始实现(手动Softmax和交叉熵)
  - PyTorch简洁实现(使用`nn.CrossEntropyLoss`)

## 🎯 学习目标

完成本章后,你将能够:
- ✅ 理解回归和分类问题的区别
- ✅ 掌握线性模型的数学原理和实现
- ✅ 理解损失函数、梯度下降的工作原理
- ✅ 熟练使用PyTorch构建和训练模型
- ✅ 处理图像分类任务和Fashion-MNIST数据集
- ✅ 评估模型性能(损失、准确率等指标)

## 📖 学习建议

1. **顺序学习**: 按01→02的顺序学习
2. **理论+实践**: 先理解数学原理,再看代码实现
3. **对比学习**: 对比"从零实现"和"简洁实现",理解PyTorch API的便利性
4. **动手实验**: 修改超参数(学习率、批量大小、训练轮数)观察效果
5. **完成练习**: 每个notebook末尾都有练习题,建议完成

## 🔑 核心知识点

### 回归 vs 分类

| 特性 | 回归 | 分类 |
|------|------|------|
| 输出类型 | 连续值 | 离散类别 |
| 例子 | 房价预测、气温预测 | 图像识别、垃圾邮件检测 |
| 损失函数 | 均方误差(MSE) | 交叉熵(Cross Entropy) |
| 激活函数 | 无/线性 | Softmax |

### 训练流程

```
初始化参数
↓
for epoch in epochs:
    for batch in dataset:
        1. 前向传播: 计算预测值
        2. 计算损失
        3. 反向传播: 计算梯度
        4. 更新参数
    评估模型性能
```

### PyTorch核心组件

- **模型**: `nn.Module`, `nn.Sequential`, `nn.Linear`
- **损失函数**: `nn.MSELoss`, `nn.CrossEntropyLoss`
- **优化器**: `torch.optim.SGD`, `torch.optim.Adam`
- **数据加载**: `torch.utils.data.DataLoader`
- **自动微分**: `loss.backward()`, `param.grad`

## 💡 常见问题

**Q1: 为什么CrossEntropyLoss不需要先Softmax?**
> PyTorch的`CrossEntropyLoss`内部已经包含了Softmax操作,直接输入logits即可,这样数值更稳定。

**Q2: 为什么训练准确率比测试准确率高?**
> 这是正常现象。模型在训练集上见过数据,自然表现更好。如果差距过大,可能是过拟合。

**Q3: 如何提高模型准确率?**
> - 增加训练轮数(但注意过拟合)
> - 调整学习率
> - 数据增强
> - 使用更复杂的模型(如多层感知机,下一章内容)

**Q4: batch_size如何选择?**
> - 太小: 训练不稳定,速度慢
> - 太大: 内存占用高,泛化能力可能下降
> - 常用: 32, 64, 128, 256

## 🚀 扩展阅读

- [PyTorch官方文档 - nn模块](https://pytorch.org/docs/stable/nn.html)
- [Fashion-MNIST数据集论文](https://arxiv.org/abs/1708.07747)
- [梯度下降优化算法综述](https://ruder.io/optimizing-gradient-descent/)

## 📊 实验结果参考

### 线性回归
- 最终损失: ~0.0003
- 收敛速度: 3-5个epoch

### Softmax回归(Fashion-MNIST)
- 训练准确率: ~85%
- 测试准确率: ~83%
- 收敛速度: 5-10个epoch

---

**下一章预告**: 多层感知机(MLP) - 引入隐藏层和非线性激活函数,突破线性模型的局限性!
