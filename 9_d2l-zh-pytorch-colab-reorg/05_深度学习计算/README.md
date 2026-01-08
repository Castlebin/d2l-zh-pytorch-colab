# 05_深度学习计算

本章介绍深度学习的计算基础,包括模型构建、参数管理和GPU加速。

## 📚 章节概览

| Notebook            | 主题      | 核心内容                       |
|---------------------|---------|----------------------------|
| 01_模型构建与参数管理.ipynb  | 模型架构与参数 | 层与块、参数管理、自定义层、模型保存         |
| 02_GPU加速计算.ipynb    | GPU计算   | GPU vs CPU、设备管理、GPU训练、多GPU |
| 03_延后初始化与参数推断.ipynb | 动态形状支持  | Lazy 模块、首次前向推断、延后初始化注意事项   |

---

## 🎯 学习目标

完成本章后,你将能够:

### 模型构建
- ✅ 理解神经网络的层和块的概念
- ✅ 使用Sequential和Module构建模型
- ✅ 实现自定义层和复杂网络架构
- ✅ 理解嵌套块和前向传播机制

### 参数管理
- ✅ 访问和查看模型参数
- ✅ 使用不同的参数初始化方法
- ✅ 实现参数共享和权重绑定
- ✅ 理解state_dict和参数管理

### 模型持久化
- ✅ 保存和加载模型参数
- ✅ 保存完整模型
- ✅ 理解.pt和.pth文件格式
- ✅ 实现检查点保存

### GPU加速
- ✅ 理解GPU vs CPU的性能差异
- ✅ 管理计算设备(CPU/GPU)
- ✅ 在GPU上训练神经网络
- ✅ 使用多GPU并行训练
- ✅ 优化GPU内存使用

### 延后初始化
- ✅ 使用 `nn.Lazy*` 模块自动推断参数形状
- ✅ 在自定义模块中实现懒惰初始化
- ✅ 理解延后初始化在保存、分布式训练中的注意事项

---

## 📖 Notebook详解

### 01_模型构建与参数管理

**Part 1: 层和块**
- nn.Module基础
- Sequential容器
- 自定义块
- 嵌套块

**Part 2: 参数管理**
- 参数访问(parameters(), named_parameters())
- state_dict使用
- 参数查看和修改

**Part 3: 参数初始化**
- 默认初始化
- Xavier初始化
- He初始化
- 自定义初始化

**Part 4: 参数共享**
- 权重绑定
- 共享层
- Siamese网络示例

**Part 5: 自定义层**
- 无参数层
- 有参数层(nn.Parameter)
- forward方法实现

**Part 6: 模型保存与加载**
- 保存/加载state_dict
- 保存/加载完整模型
- 检查点保存

**关键代码**:
```python
# 自定义块
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(256, 128)
        self.output = nn.Linear(128, 10)
    
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

# 参数初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

net.apply(init_weights)

# 保存和加载
torch.save(net.state_dict(), 'model.pt')
net.load_state_dict(torch.load('model.pt'))
```

### 02_GPU加速计算

**Part 1: GPU基础知识**
- GPU vs CPU对比
- CUDA和cuDNN
- 查看GPU信息

**Part 2: 计算设备**
- torch.device使用
- try_gpu()辅助函数
- 设备管理

**Part 3: 张量与GPU**
- 在GPU上创建张量
- CPU-GPU数据传输
- 设备间复制

**Part 4: 性能对比**
- 矩阵乘法性能测试
- 不同规模的加速比
- 可视化对比

**Part 5: 神经网络与GPU**
- 模型移动到GPU
- GPU上训练
- 训练速度对比

**Part 6: 多GPU训练**
- DataParallel使用
- 数据并行原理

**Part 7: 最佳实践**
- 常见陷阱
- 标准训练模板
- 性能优化

**关键代码**:
```python
# 设备管理
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 移动模型和数据
model = model.to(device)
data, target = data.to(device), target.to(device)

# 多GPU训练
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 性能监控
torch.cuda.synchronize()
print(f'Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB')
```

---

### 03_延后初始化与参数推断

**Part 1: 懒惰初始化动机**
- 动态输入维度的建模挑战
- 手动指定 vs 自动推断的比较

**Part 2: PyTorch Lazy 模块**
- `nn.LazyLinear`, `nn.LazyConv2d`, `nn.LazyBatchNorm`
- 观察初始化前后的权重形状

**Part 3: 自定义延后初始化模块**
- 在 `forward` 中按需创建层
- GPU/多卡训练的同步技巧
- `state_dict` 保存、加载与断点恢复

**关键代码**:
```python
lazy_net = nn.Sequential(
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.LazyLinear(10)
)

# 第一次前向传播时自动推断 in_features
output = lazy_net(torch.randn(16, 64))
```

---

## 🔑 核心概念

### 1. 层和块

**层(Layer)**: 最小的计算单元
```python
linear = nn.Linear(10, 5)  # 全连接层
conv = nn.Conv2d(3, 64, 3)  # 卷积层
```

**块(Block)**: 多个层的组合
```python
class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)
    
    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))
```

### 2. nn.Module vs nn.Sequential

| nn.Module      | nn.Sequential |
|----------------|---------------|
| 灵活,可自定义forward | 简单,自动串联       |
| 适合复杂网络         | 适合简单网络        |
| 可添加控制流         | 只能顺序执行        |
| 需要实现forward    | 不需要forward    |

### 3. 参数初始化方法

| 方法     | 适用场景          | 公式                                                                |
|--------|---------------|-------------------------------------------------------------------|
| Xavier | sigmoid, tanh | $W \sim U[-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})}]$ |
| He     | ReLU          | $W \sim N(0, \sqrt{2/n_{in}})$                                    |
| 常数     | Bias          | $b = 0$                                                           |

### 4. GPU计算优势

| 操作             | CPU | GPU | 加速比      |
|----------------|-----|-----|----------|
| 小矩阵(100×100)   | 快   | 慢   | ~0.5x    |
| 中矩阵(1000×1000) | 中   | 快   | ~5x      |
| 大矩阵(4000×4000) | 慢   | 很快  | ~20x     |
| 深度学习训练         | 慢   | 很快  | ~10-100x |

---

## 💡 重点提示

### 模型构建
1. **使用Sequential**: 简单网络用Sequential
2. **自定义Module**: 复杂网络继承nn.Module
3. **模块化设计**: 将重复结构封装成块
4. **命名规范**: 使用有意义的层名称

### 参数管理
1. **访问参数**: 使用`named_parameters()`遍历
2. **修改参数**: 使用`with torch.no_grad()`
3. **初始化时机**: 在训练前初始化
4. **保存best模型**: 根据验证集性能保存

### GPU使用
1. **设备一致性**: 模型和数据必须在同一设备
2. **减少传输**: 避免频繁的CPU-GPU传输
3. **内存管理**: 监控GPU内存使用
4. **批量大小**: GPU适合更大的batch_size

---

## 🎓 学习建议

### 第一轮学习(理解概念)
1. ⏰ **时间**: 3-4小时
2. 📝 **目标**: 
   - 理解层、块、模型的概念
   - 掌握参数访问和初始化
   - 了解GPU的基本使用
3. 💻 **实践**: 
   - 运行所有代码单元
   - 修改参数观察效果
   - 测试GPU vs CPU性能

### 第二轮学习(深入实践)
1. ⏰ **时间**: 2-3小时
2. 📝 **目标**:
   - 实现复杂的自定义模型
   - 掌握多种初始化方法
   - 编写GPU训练代码
3. 💻 **实践**:
   - 完成文末练习
   - 实现自己的网络架构
   - 在GPU上训练完整模型

### 第三轮学习(优化提升)
1. ⏰ **时间**: 2小时
2. 📝 **目标**:
   - 优化模型结构
   - 改进训练性能
   - 掌握最佳实践
3. 💻 **实践**:
   - 性能调优
   - 内存优化
   - 多GPU训练

---

## 📊 知识点检查清单

### 基础知识(必须掌握)
- [ ] 能够使用Sequential构建简单模型
- [ ] 能够继承nn.Module创建自定义模型
- [ ] 理解forward方法的作用
- [ ] 能够访问和查看模型参数
- [ ] 能够保存和加载模型
- [ ] 知道如何检查GPU是否可用
- [ ] 能够将模型和数据移动到GPU

### 进阶知识(建议掌握)
- [ ] 能够实现嵌套的复杂网络
- [ ] 能够实现参数共享
- [ ] 能够创建自定义层(有参数和无参数)
- [ ] 能够使用不同的初始化方法
- [ ] 能够编写设备无关的训练代码
- [ ] 能够进行GPU性能测试
- [ ] 理解DataParallel的工作原理

### 高级知识(深入理解)
- [ ] 能够实现复杂的参数初始化策略
- [ ] 能够优化模型架构提高性能
- [ ] 能够实现高效的检查点保存
- [ ] 能够优化GPU内存使用
- [ ] 能够使用混合精度训练
- [ ] 能够进行多GPU分布式训练
- [ ] 能够分析和解决GPU性能瓶颈

---

## 🔧 实战练习

### 练习1: 自定义ResNet块 ⭐⭐
实现一个残差块,包含跳跃连接。

**提示**:
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: 实现主路径
        # TODO: 实现跳跃连接
    
    def forward(self, x):
        # TODO: 实现残差连接
        pass
```

### 练习2: 参数统计 ⭐
编写函数统计模型的参数数量和大小。

```python
def count_parameters(model):
    # TODO: 统计总参数数
    # TODO: 统计可训练参数数
    # TODO: 计算模型大小(MB)
    pass
```

### 练习3: 智能设备管理 ⭐⭐
实现一个设备管理器,自动选择最优设备。

```python
class DeviceManager:
    def __init__(self):
        # TODO: 检测所有可用设备
        # TODO: 选择最优设备
        pass
    
    def to(self, *args):
        # TODO: 移动所有参数到设备
        pass
```

### 练习4: GPU性能分析 ⭐⭐⭐
在不同batch_size下测试GPU利用率。

**要求**:
- 测试batch_size从32到512
- 绘制GPU利用率曲线
- 找出最优batch_size

### 练习5: 混合精度训练 ⭐⭐⭐
实现FP16混合精度训练。

**提示**: 使用`torch.cuda.amp`

### 练习6: 多模型集成 ⭐⭐⭐
实现一个训练多个模型并集成预测的系统。

**要求**:
- 训练3个不同架构的模型
- 保存每个模型的checkpoint
- 实现投票或平均集成

---

## 🐛 常见问题

### Q1: RuntimeError: Expected all tensors to be on the same device
**原因**: 模型在GPU,数据在CPU,或反之。

**解决**:
```python
device = torch.device('cuda')
model = model.to(device)
data = data.to(device)  # 别忘了移动数据!
```

### Q2: RuntimeError: CUDA out of memory
**原因**: GPU内存不足。

**解决方案**:
```python
# 方案1: 减小batch_size
batch_size = 32  # 改为16或更小

# 方案2: 清理缓存
torch.cuda.empty_cache()

# 方案3: 使用梯度累积
for i, (data, target) in enumerate(loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q3: 为什么小模型GPU比CPU慢?
**原因**: 数据传输开销大于计算收益。

**解释**:
- CPU-GPU传输有延迟
- 小数据无法充分利用GPU并行能力
- GPU适合大规模并行计算

### Q4: 如何查看GPU使用情况?
**方法**:
```bash
# 方法1: nvidia-smi
nvidia-smi

# 方法2: watch命令实时查看
watch -n 1 nvidia-smi

# 方法3: Python代码
import torch
print(torch.cuda.memory_allocated() / 1e9)  # GB
```

### Q5: 保存的模型可以在没有GPU的机器上加载吗?
**可以!** 使用map_location:
```python
# 在CPU上加载GPU训练的模型
model.load_state_dict(torch.load('model.pt', map_location='cpu'))

# 在GPU上加载CPU训练的模型
model.load_state_dict(torch.load('model.pt', map_location='cuda'))
```

### Q6: DataParallel和DistributedDataParallel有什么区别?
| DataParallel | DistributedDataParallel |
|--------------|-------------------------|
| 单进程多线程 | 多进程 |
| 简单易用 | 配置复杂 |
| 速度较慢 | 速度更快 |
| 适合单机多卡 | 适合多机多卡 |

---

## 🚀 进阶资源

### 推荐阅读
1. **PyTorch官方文档**: 
   - [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
   - [CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)

2. **性能优化**:
   - [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
   - [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

3. **分布式训练**:
   - [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### 下一步学习
完成本章后,建议学习:
- **Chapter 6**: 卷积神经网络(CNN)
- **Chapter 7**: 现代卷积神经网络(ResNet, VGG等)
- **Chapter 8**: 循环神经网络(RNN)

---

## 📝 笔记模板

```markdown
# 我的学习笔记 - 深度学习计算

## 学习日期
- 开始: YYYY-MM-DD
- 完成: YYYY-MM-DD

## 重要概念
1. **层和块**:
   - 我的理解: [写下你的理解]
   - 关键代码: [粘贴重要代码]

2. **参数初始化**:
   - Xavier适用于: [...]
   - He适用于: [...]

3. **GPU使用**:
   - 我的设备: [...]
   - 性能提升: [...]倍

## 遇到的问题
1. **问题**: [描述问题]
   **解决**: [解决方案]

## 练习完成情况
- [ ] 练习1: 自定义ResNet块
- [ ] 练习2: 参数统计
- [ ] 练习3: 智能设备管理
- [ ] 练习4: GPU性能分析
- [ ] 练习5: 混合精度训练
- [ ] 练习6: 多模型集成

## 代码片段
```python
# 我觉得最有用的代码
[粘贴代码]
```

## 下次学习计划
- [ ] 复习不理解的概念
- [ ] 完成剩余练习
- [ ] 开始下一章学习
```

---

## 🎉 总结

本章是深度学习工程化的基础,掌握这些技能将使你能够:
- 🏗️ 构建任意复杂度的神经网络
- 🎯 灵活管理模型参数
- 💾 正确保存和加载模型
- ⚡ 充分利用GPU加速训练
- 🔧 优化训练性能

**核心思想**: 
> "好的模型设计 + 正确的参数管理 + 高效的GPU使用 = 成功的深度学习项目"

继续加油!💪 下一章我们将学习**卷积神经网络**,进入计算机视觉的世界!🖼️
