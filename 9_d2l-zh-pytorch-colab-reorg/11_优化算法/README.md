# 第10章: 优化算法

本章全面介绍深度学习中的优化算法,从基础梯度下降到现代自适应方法。

## 📚 内容概览

### [01_优化基础与梯度下降](./01_优化基础与梯度下降.ipynb)
**优化理论基础与经典算法**

#### 核心内容
1. **优化 vs 深度学习**
   - 训练误差(经验风险) vs 泛化误差(真实风险)
   - 过拟合问题
   - 优化挑战: 局部最小值、鞍点、梯度消失

2. **凸性理论**
   - 凸集与凸函数定义
   - 詹森不等式(Jensen's Inequality)
   - 凸优化的优势: 局部最小值=全局最小值

3. **梯度下降(GD)**
   - 泰勒展开推导
   - 学习率的影响(太小/太大)
   - 一维和多维实现

4. **随机梯度下降(SGD)**
   - 计算效率: O(1) vs O(n)
   - 无偏梯度估计
   - 噪声的利弊

5. **小批量SGD**
   - 批量大小的权衡
   - 向量化与并行效率
   - 实践中的标准选择

#### 关键公式
```
梯度下降: w_t = w_{t-1} - η∇f(w_{t-1})
SGD: w_t = w_{t-1} - η∇f_i(w_{t-1})  # 随机采样i
Mini-batch: w_t = w_{t-1} - (η/b)Σ∇f_i(w_{t-1})  # 批量b
```

---

### [02_动量方法与自适应学习率](./02_动量方法与自适应学习率.ipynb)
**现代优化算法的两大改进方向**

#### 核心内容
1. **Momentum(动量法)**
   - 指数加权移动平均
   - 物理类比: 球滚下山坡
   - 解决病态条件问题(Ill-conditioned)
   - 减少振荡,加速收敛

2. **AdaGrad**
   - 稀疏特征问题
   - 自适应学习率机制
   - 累积梯度平方和
   - 问题: 学习率单调递减

3. **RMSProp**
   - 解决AdaGrad衰减过快
   - 梯度平方的指数移动平均
   - 更适合非凸优化

4. **Adam**
   - 结合Momentum和RMSProp
   - 一阶矩和二阶矩估计
   - 偏差修正(Bias Correction)
   - 深度学习最流行优化器

5. **Adadelta**
   - 无学习率参数
   - 参数变化量的自适应缩放
   - 理论有趣但实践较少

#### 算法对比表

| 优化器      | 核心机制             | 状态变量  | 超参数       | 适用场景      |
|----------|------------------|-------|-----------|-----------|
| SGD      | 负梯度              | 无     | η         | 凸优化,需精细调参 |
| Momentum | 累积梯度             | v     | η, β      | 加速,平滑     |
| AdaGrad  | 自适应LR            | s     | η         | 稀疏数据      |
| RMSProp  | 梯度平方EMA          | s     | η, γ      | RNN,非凸    |
| **Adam** | Momentum+RMSProp | v, s  | η, β₁, β₂ | **通用首选**  |
| Adadelta | 无LR              | s, Δx | ρ         | 理论研究      |

#### Adam公式
```python
v_t = β₁v_{t-1} + (1-β₁)g_t          # 一阶矩
s_t = β₂s_{t-1} + (1-β₂)g_t²         # 二阶矩
v̂_t = v_t / (1 - β₁^t)               # 偏差修正
ŝ_t = s_t / (1 - β₂^t)
w_t = w_{t-1} - η·v̂_t / (√ŝ_t + ε)  # 更新

# 默认超参数
β₁ = 0.9, β₂ = 0.999, ε = 1e-8, η = 1e-3
```

---

### [03_学习率调度与实践技巧](./03_学习率调度与实践技巧.ipynb)
**优化的最佳实践与高级技巧**

#### 核心内容
1. **学习率调度策略**
   - **Step Decay**: 阶梯式衰减
   - **Exponential Decay**: 指数衰减
   - **Polynomial Decay**: 多项式衰减(√t)
   - **Cosine Annealing**: 余弦退火(推荐)
   - **Warmup**: 预热策略

2. **PyTorch调度器**
   - `StepLR`: 固定间隔衰减
   - `ExponentialLR`: 指数衰减
   - `CosineAnnealingLR`: 余弦退火
   - `ReduceLROnPlateau`: 基于指标自适应
   - `OneCycleLR`: 循环学习率

3. **批量大小选择**
   - 小批量(32-64): 泛化好,噪声大
   - 大批量(512+): 稳定,可能泛化差
   - 线性缩放规则: Batch增k倍 → LR增k倍

4. **训练技巧**
   - 梯度裁剪(防止梯度爆炸)
   - 权重衰减(L2正则化)
   - Early Stopping
   - 混合精度训练

#### 学习率调度公式

```python
# 1. Step Decay
lr_t = lr_0 × γ^⌊t/T⌋

# 2. Exponential Decay
lr_t = lr_0 × γ^t

# 3. Polynomial Decay
lr_t = lr_0 / √(t+1)

# 4. Cosine Annealing
lr_t = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))

# 5. Warmup + Cosine
if t < T_warmup:
    lr_t = (t / T_warmup) × lr_0
else:
    lr_t = Cosine(t - T_warmup)
```

---

## 🎯 优化器选择速查表

### 按任务类型

| 任务                  | 推荐优化器        | 配置                              |
|---------------------|--------------|---------------------------------|
| **通用/默认**           | Adam         | `lr=1e-3, β₁=0.9, β₂=0.999`     |
| **计算机视觉**           | SGD+Momentum | `lr=0.1, momentum=0.9, wd=1e-4` |
| **Transformer/NLP** | AdamW        | `lr=5e-5, warmup+cosine`        |
| **稀疏数据**            | AdaGrad      | `lr=0.01`                       |
| **RNN**             | RMSProp      | `lr=1e-3, α=0.9`                |

### 按训练阶段

| 阶段     | 策略             | 原因     |
|--------|----------------|--------|
| **初期** | 大学习率 or Warmup | 快速接近最优 |
| **中期** | 适中学习率          | 稳定下降   |
| **后期** | 小学习率           | 精细收敛   |

---

## 💡 最佳实践建议

### 1. 默认配置(快速开始)
```python
# 适用于大多数任务
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)

batch_size = 64
```

### 2. 计算机视觉(追求最优性能)
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True  # Nesterov动量
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 60, 90],  # epochs
    gamma=0.1
)

batch_size = 128
```

### 3. Transformer/大模型
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# Warmup + Linear Decay
from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)

batch_size = 32
```

---

## 📊 训练诊断与调试

### 常见问题速查

| 问题        | 可能原因       | 解决方案            |
|-----------|------------|-----------------|
| Loss不下降   | LR太小       | 增大学习率(×10)      |
| Loss震荡/发散 | LR太大       | 减小学习率(÷10),梯度裁剪 |
| 训练后期停滞    | LR太大       | 添加学习率衰减         |
| GPU利用率低   | Batch太小    | 增大batch,缩放LR    |
| 泛化性能差     | Batch太大    | 减小batch,加强正则化   |
| 梯度爆炸(NaN) | LR太大或网络不稳定 | 梯度裁剪,降低LR       |

### 调试流程

```
1. 小数据过拟合测试(100样本)
   ↓
2. 全数据,固定LR找合适初始学习率
   ↓
3. 添加学习率调度提升性能
   ↓
4. 调整批量大小优化速度
   ↓
5. 添加正则化改善泛化
```

---

## 🔬 理论深入

### 为什么Adam这么流行?

1. **自动调整**: 不同参数不同学习率
2. **鲁棒性**: 对超参数不敏感
3. **快速收敛**: 结合动量和自适应的优点
4. **广泛验证**: 在各种任务上表现稳定

### 动量法的本质

```
速度更新: v_t = βv_{t-1} + g_t

展开: v_t = g_t + βg_{t-1} + β²g_{t-2} + ...
           = Σ β^τ g_{t-τ}

效果:
- 平滑梯度噪声
- 加速一致方向
- 减缓振荡方向
```

### 自适应学习率的本质

```
更新: w_t = w_{t-1} - η / √s_t × g_t

其中: s_t ∝ 过去梯度的平方

效果:
- 梯度大的维度 → s_t大 → 学习率小
- 梯度小的维度 → s_t小 → 学习率大
- 自动平衡不同方向的更新步长
```

---

## 🔧 实用工具

### 学习率查找器
```python
def find_lr(model, train_loader, init_lr=1e-8, final_lr=10):
    """自动寻找最优学习率"""
    optimizer = optim.SGD(model.parameters(), lr=init_lr)
    lr_scheduler = ExponentialLR(optimizer, gamma=1.1)
    
    lrs, losses = [], []
    for batch in train_loader:
        loss = train_step(model, batch, optimizer)
        losses.append(loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        
        lr_scheduler.step()
        if optimizer.param_groups[0]['lr'] > final_lr:
            break
    
    # 绘制loss vs lr曲线,选择下降最快的lr
    plt.plot(lrs, losses)
    plt.xscale('log')
```

### 梯度裁剪
```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0  # 最大梯度范数
)
```

### Early Stopping
```python
best_val_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(num_epochs):
    val_loss = validate(model)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(model)
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("Early stopping!")
        break
```

---

## 📖 推荐阅读

### 经典论文
1. **Momentum**: Polyak (1964) - "Some methods of speeding up the convergence of iteration methods"
2. **AdaGrad**: Duchi et al. (2011) - "Adaptive Subgradient Methods"
3. **RMSProp**: Hinton (2012) - Coursera Lecture 6e
4. **Adam**: Kingma & Ba (2014) - "Adam: A Method for Stochastic Optimization"
5. **Learning Rate Schedules**: Smith (2017) - "Cyclical Learning Rates"

### 在线资源
- [PyTorch优化器文档](https://pytorch.org/docs/stable/optim.html)
- [CS231n优化笔记](http://cs231n.github.io/neural-networks-3/)
- [Sebastian Ruder博客: 优化算法概览](https://ruder.io/optimizing-gradient-descent/)

---

## ⚡ 快速参考

### Adam完整代码
```python
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in params]  # 一阶矩
        self.v = [torch.zeros_like(p) for p in params]  # 二阶矩
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            
            # 更新动量
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            
            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # 参数更新
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
```

### 训练循环模板
```python
for epoch in range(num_epochs):
    # 训练
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # 验证
    model.eval()
    val_loss = validate(model, val_loader)
    
    # 学习率调度
    scheduler.step()  # 或 scheduler.step(val_loss) for ReduceLROnPlateau
    
    # 记录
    print(f"Epoch {epoch}: Loss={val_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
```

---

## 🎓 练习建议

### 基础练习
1. 在简单函数(如$f(x)=x^2$)上实现并可视化GD, SGD, Momentum, Adam
2. 比较不同学习率(0.001, 0.01, 0.1, 1.0)的收敛曲线
3. 验证线性缩放规则: batch翻倍,lr翻倍

### 进阶练习
4. 在MNIST上对比5种优化器(SGD, Momentum, AdaGrad, RMSProp, Adam)
5. 实现并测试5种学习率调度策略
6. 实现学习率查找器,自动找最优学习率

### 高级挑战
7. 在CIFAR-10上复现ResNet训练,应用所有最佳实践
8. 实现AdamW(Adam with decoupled weight decay)
9. 研究并实现Lookahead或RAdam等新型优化器

---

## 💬 常见疑问

**Q: 为什么Adam这么流行但很多论文用SGD+Momentum?**  
A: Adam收敛快,适合快速实验。SGD+Momentum需要精细调参,但最终性能可能更好(特别是CV任务)。

**Q: 学习率应该从多大开始?**  
A: Adam默认1e-3, SGD默认0.1。使用学习率查找器或从小开始(1e-5)倍增测试。

**Q: 什么时候需要Warmup?**  
A: 1) 使用大批量(>512); 2) 训练Transformer; 3) 学习率很大时。

**Q: 如何选择批量大小?**  
A: 从32/64开始,根据GPU内存增大。大批量需要缩放学习率和Warmup。

**Q: 梯度裁剪什么时候用?**  
A: 训练RNN/Transformer或出现梯度爆炸(loss突然变NaN)时必须使用。

---

**总结**: 优化算法是深度学习成功的关键。掌握本章内容后,你将能够为任何任务选择合适的优化策略,并有效调试训练过程!
