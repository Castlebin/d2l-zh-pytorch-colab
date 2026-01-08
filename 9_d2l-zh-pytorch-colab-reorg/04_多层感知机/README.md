# 04 多层感知机

本章介绍深度学习的基础 - **多层感知机(MLP)**,以及训练深度神经网络的关键技术。

## 📚 学习内容

### 1. 多层感知机与反向传播 (01_多层感知机与反向传播.ipynb)
- **为什么需要MLP**: 突破线性模型的局限
- **XOR问题**: 线性不可分的经典案例
- **激活函数**: ReLU, Sigmoid, Tanh的原理与对比
- **从零实现MLP**: 手动实现两层神经网络
- **反向传播**: 链式法则与梯度计算
- **梯度消失/爆炸**: 深层网络的挑战

### 2. 模型选择与正则化 (02_模型选择与正则化.ipynb)
- **过拟合与欠拟合**: 机器学习的核心问题
- **训练误差vs泛化误差**: 评估模型的真正能力
- **权重衰减(L2正则化)**: 限制模型复杂度
- **Dropout暂退法**: 随机丢弃神经元防止过拟合
- **参数初始化**: Xavier和He初始化方法
- **数值稳定性**: 梯度消失与爆炸的解决方案

### 3. Kaggle实战: 房价预测 (03_Kaggle实战_房价预测.ipynb) ⭐⭐⭐
- **真实数据处理**: 混合数据类型的预处理
- **特征工程**: 缺失值处理、标准化、独热编码
- **K折交叉验证**: 模型选择和超参数调优
- **Kaggle竞赛**: 完整的竞赛流程实践
- **对数损失**: 相对误差的优化技巧
- **结果提交**: 生成CSV并上传到Kaggle平台

### 4. 分布偏移与部署挑战 (04_分布偏移与部署挑战.ipynb) ⭐⭐
- **分布偏移类型**: 协变量偏移、先验偏移、概念漂移
- **线上风险**: 模型上线后性能骤降的典型原因
- **监控指标**: 数据统计、预测置信度、漂移检测算法
- **缓解策略**: 数据增强、领域自适应、在线学习、模型集成
- **小实验**: 用 PyTorch 构造协变量偏移并观察准确率下降

## 🎯 学习目标

完成本章后,你将能够:
- ✅ 理解为什么需要隐藏层和激活函数
- ✅ 实现和训练多层感知机
- ✅ 识别和解决过拟合问题
- ✅ 应用权重衰减和Dropout正则化技术
- ✅ 理解反向传播的工作原理
- ✅ 选择合适的参数初始化方法
- ✅ 构建泛化能力强的深度模型
- ✅ 参加Kaggle竞赛并提交预测结果
- ✅ 识别并监控分布偏移,制定上线后的维护策略

## 📖 学习建议

1. **顺序学习**: 按01→02→03的顺序学习
2. **动手实验**: 
   - 修改隐藏层大小观察效果
   - 尝试不同的dropout概率
   - 对比不同的激活函数
   - 调整Kaggle竞赛的超参数
3. **理解原理**: 
   - 手动推导反向传播公式
   - 理解为什么Dropout有效
   - 学习K折交叉验证的作用
   - 结合第04节思考线上线下评估的差异
4. **可视化**: 观察训练/测试误差曲线判断拟合情况
5. **实践为主**: 
   - 在Fashion-MNIST上反复实验
   - 尝试提升Kaggle排名

## 🔑 核心知识点

### MLP vs 线性模型

| 特性               | 线性模型  | 多层感知机  |
|------------------|-------|--------|
| 隐藏层              | 无     | 有      |
| 激活函数             | 无     | ReLU等  |
| 参数量              | 少     | 多      |
| 表达能力             | 弱(线性) | 强(非线性) |
| 过拟合风险            | 低     | 高      |
| Fashion-MNIST准确率 | ~83%  | ~88%   |

### 激活函数对比

| 激活函数    | 公式                | 优点        | 缺点     | 适用场景     |
|---------|-------------------|-----------|--------|----------|
| ReLU    | max(0,x)          | 简单,缓解梯度消失 | 死亡ReLU | 现代网络首选   |
| Sigmoid | 1/(1+e⁻ˣ)         | 输出(0,1)   | 梯度消失严重 | 输出层(二分类) |
| Tanh    | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 零中心化      | 梯度消失   | RNN中使用   |

### 正则化技术对比

| 方法             | 原理      | 优点    | 缺点    | 实现                     |
|----------------|---------|-------|-------|------------------------|
| 权重衰减           | 惩罚大权重   | 简单有效  | 需要调λ  | `weight_decay=λ`       |
| Dropout        | 随机丢弃神经元 | 效果显著  | 训练慢2倍 | `nn.Dropout(p)`        |
| 数据增强           | 扩充训练数据  | 提升泛化  | 特定领域  | torchvision.transforms |
| Early Stopping | 早停      | 防止过训练 | 需要验证集 | 手动实现                   |

### 训练流程

```python
# 1. 构建模型
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),      # Dropout正则化
    nn.Linear(256, 10)
)

# 2. 定义损失和优化器
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), 
    lr=0.1, 
    weight_decay=0.01    # L2正则化
)

# 3. 训练循环
for epoch in range(num_epochs):
    for X, y in train_iter:
        optimizer.zero_grad()
        l = loss(net(X), y)
        l.backward()         # 反向传播
        optimizer.step()     # 更新参数
```

## 💡 常见问题

**Q1: 隐藏层应该有多少个神经元?**
> 经验法则: 从256或512开始,根据任务复杂度调整。太少可能欠拟合,太多容易过拟合。

**Q2: 何时使用Dropout?**
> 当观察到训练准确率远高于测试准确率时(过拟合的标志),应该加入Dropout。通常p=0.5是个好的起点。

**Q3: 权重衰减系数λ如何选择?**
> 常用值: 0.001, 0.01, 0.1。从小开始尝试,观察测试误差。λ太大会导致欠拟合。

**Q4: ReLU vs Sigmoid选哪个?**
> 几乎总是选ReLU! Sigmoid容易梯度消失,只在输出层(二分类)时使用。

**Q5: 如何判断是否过拟合?**
> 看训练误差和测试误差的差距:
> - 差距小: 拟合良好
> - 差距大: 过拟合,需要正则化
> - 都很高: 欠拟合,需要更复杂的模型

**Q6: 为什么需要参数初始化?**
> 全零初始化会导致对称性问题(所有神经元学到相同的特征)。随机初始化打破对称性,但要注意方差,防止梯度消失/爆炸。

## 🔬 实验建议

### 实验1: 观察过拟合
```python
# 小数据集 + 大模型 = 过拟合
# 使用前1000个样本训练
train_iter_small = DataLoader(mnist_train[:1000], batch_size)
# 使用5层网络
net_large = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 10)
)
# 观察训练/测试误差的差距
```

### 实验2: Dropout效果
```python
# 对比有无Dropout的测试准确率
net_no_dropout = nn.Sequential(..., nn.Linear(256, 256), nn.ReLU(), ...)
net_with_dropout = nn.Sequential(..., nn.Dropout(0.5), nn.Linear(256, 256), nn.ReLU(), ...)
```

### 实验3: 学习率调度
```python
# 学习率衰减
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
for epoch in range(num_epochs):
    train(...)
    scheduler.step()  # 每 5 个 epoch 学习率减半
```

## 🚀 扩展阅读

- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html) - Xavier初始化论文
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) - He初始化论文
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) - Dropout原论文
- [Batch Normalization](https://arxiv.org/abs/1502.03167) - 另一种正则化技术

## 📊 性能参考

### Fashion-MNIST分类

| 模型                    | 测试准确率  | 训练时间(10 epochs) |
|-----------------------|--------|-----------------|
| Softmax回归             | ~83%   | ~1分钟            |
| MLP(1层,256)           | ~88%   | ~2分钟            |
| MLP(1层,256) + Dropout | ~87%   | ~3分钟            |
| MLP(2层,256)           | ~88.5% | ~3分钟            |
| MLP(2层,256) + Dropout | ~87.5% | ~4分钟            |

*注: Dropout短期可能降低准确率,但显著提升泛化能力*

## 🎓 下一步

完成本章后,你已经掌握了深度学习的核心基础! 

**下一章预告**: 深度学习计算 - 学习如何:
- 构建更复杂的模型架构
- 管理和保存模型参数
- 自定义层和模块
- 使用GPU加速训练

继续加油! 🚀
