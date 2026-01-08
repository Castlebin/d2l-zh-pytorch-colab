# Chapter 11: 计算性能 (Computational Performance)

深度学习的计算性能优化是实践中的关键话题。本章介绍硬件基础、异步计算、并行化策略和多GPU训练技术。

## 📚 内容概览

### Notebook 1: 硬件与异步计算
**文件**: `01_硬件与异步计算.ipynb`

**主要内容**:
- **硬件基础**: CPU/GPU架构、内存层次、存储系统
- **性能指标**: 延迟、带宽、吞吐量
- **CPU详解**: 缓存层次、向量化(SIMD)、多核并行
- **GPU详解**: SM架构、Tensor Core、HBM显存
- **异步计算**: 前端/后端分离、依赖跟踪、同步点
- **性能优化**: 内存访问模式、异步传输、流水线

**关键概念**:
- L1/L2/L3缓存层次 (ns级 → μs级)
- CPU向量化: AVX2/AVX-512 (8-16×加速)
- GPU并行: 数千核心,TB/s带宽
- 异步执行: 计算与通信重叠
- `non_blocking=True`: 异步数据传输

**代码示例**:
```python
# 异步传输
data = data.to(device, non_blocking=True)

# 计算与通信重叠
for i in range(n):
    result = compute_on_gpu(data[i])
    result_cpu = result.to('cpu', non_blocking=True)  # 异步
torch.cuda.synchronize()  # 批量同步
```

---

### Notebook 2: 自动并行与多GPU基础
**文件**: `02_自动并行与多GPU基础.ipynb`

**主要内容**:
- **自动并行化**: 框架自动调度、依赖分析
- **多设备并行**: CPU+GPU、多GPU并行
- **计算通信重叠**: 资源分离、Pipeline
- **三种并行策略**: 网络并行、层内并行、数据并行
- **数据并行详解**: 工作流、实现方法
- **参数同步**: All-Reduce vs Parameter Server

**三种并行对比**:

| 策略       | 原理              | 优点       | 缺点        | 推荐度        |
|----------|-----------------|----------|-----------|------------|
| 网络并行     | 不同层在不同GPU       | -        | 频繁通信,利用率低 | ❌ 不推荐      |
| 层内并行     | 通道分配到不同GPU      | -        | 每层都需同步    | ❌ 不推荐      |
| **数据并行** | 数据分批,各GPU处理不同数据 | 简单、高效、通用 | 需聚合梯度     | ✅ **强烈推荐** |

**数据并行工作流**:
```
1. 数据分割 → batch拆成k份 (k=GPU数)
2. 并行前向 → 每GPU独立计算
3. 并行反向 → 每GPU计算局部梯度
4. 梯度聚合 → All-Reduce求平均
5. 参数更新 → 每GPU同步更新
```

**PyTorch实现**:
```python
# 方法1: nn.DataParallel (简单)
model = nn.DataParallel(model)
model = model.to('cuda:0')

# 方法2: DistributedDataParallel (高效)
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

**批大小与学习率调整**:
```
k个GPU:
  - 批大小: base_batch × k
  - 学习率: base_lr × k (通常)
  - 可能需要warmup
```

---

### Notebook 3: 多GPU训练实战
**文件**: `03_多GPU训练实战.ipynb`

**主要内容**:
- **从零实现**: 手写数据并行(get_params, allreduce, split_batch)
- **LeNet多GPU训练**: 完整实现示例
- **PyTorch高级API**: DataParallel vs DDP
- **ResNet-18示例**: 大模型多GPU训练
- **参数服务器**: Push-Pull模型、多服务器架构
- **性能优化**: 梯度累积、混合精度、梯度压缩
- **调试技巧**: 数值验证、Profiling、常见问题

**从零实现核心代码**:
```python
def train_batch(X, y, device_params, devices, lr, net, loss_fn):
    """完整的多GPU训练步骤"""
    # 1. 分割数据
    X_shards, y_shards = split_batch(X, y, devices)
    
    # 2-3. 并行前向+反向
    losses = []
    for X_shard, y_shard, params in zip(X_shards, y_shards, device_params):
        y_hat = net(X_shard, params)
        l = loss_fn(y_hat, y_shard).sum()
        l.backward()
        losses.append(l)
    
    # 4. All-Reduce梯度
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad 
                      for c in range(len(devices))])
    
    # 5. 更新参数
    for params in device_params:
        for param in params:
            param[:] -= lr * param.grad / X.shape[0]
            param.grad.zero_()
```

**参数服务器架构**:
```
┌─────────────┐
│  Parameter  │ ← 存储参数、聚合梯度
│   Server    │ ← 分发更新
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
Worker  Worker ← 计算梯度
GPU 0   GPU 1  ← 前向/反向传播

工作流:
1. Pull: 拉取参数
2. Compute: 计算梯度  
3. Push: 推送梯度
4. Aggregate: 聚合并更新
```

**性能优化技巧**:

1. **梯度累积** (大批量训练)
```python
accumulation_steps = 4
for i, (X, y) in enumerate(dataloader):
    loss = model(X, y) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

2. **混合精度** (FP16加速)
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for X, y in dataloader:
    with autocast():
        output = model(X)
        loss = criterion(output, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

3. **通信优化**
```python
model = DDP(model, 
           bucket_cap_mb=25,  # 分桶传输
           gradient_as_bucket_view=True)  # 优化内存
```

---

## 🎯 核心概念总结

### 1. 硬件性能层次
```
速度: 寄存器 > L1 > L2 > L3 > RAM > SSD > HDD > 网络
差距: 每级慢10-1000倍

优化原则:
  - 利用缓存局部性
  - 减少内存访问
  - 批处理操作
  - 异步执行
```

### 2. 异步计算模型
```
前端(Python)         后端(C++)
    │                   │
提交操作 ──────────> 执行队列
    │                   │
继续执行             并行计算
    │                   │
需要结果 <────同步──── 完成

关键API:
  - .to(device, non_blocking=True)
  - torch.cuda.synchronize()
  - torch.cuda.stream()
```

### 3. 数据并行最佳实践
```
配置:
  ✅ batch_size = base_batch × GPU数
  ✅ lr = base_lr × GPU数 (可能需调整)
  ✅ pin_memory=True
  ✅ non_blocking=True
  ✅ 充足的num_workers

实现选择:
  单机2-4卡  → nn.DataParallel (简单)
  单机多卡   → DistributedDataParallel (高效)
  多机训练   → DDP + NCCL
```

### 4. 同步策略对比

| 方法               | 通信模式 | 复杂度    | 扩展性 | 推荐场景     |
|------------------|------|--------|-----|----------|
| **All-Reduce**   | 点对点  | O(1)   | 优秀  | 单机多GPU   |
| Parameter Server | 中心化  | O(m/n) | 良好  | 多机训练     |
| Ring All-Reduce  | 环形   | O(1)   | 优秀  | NVLink拓扑 |

### 5. 性能优化Checklist

**数据加载**:
- [ ] pin_memory=True
- [ ] num_workers=4×GPU数
- [ ] prefetch_factor=2

**计算优化**:
- [ ] 混合精度训练(AMP)
- [ ] 梯度检查点(大模型)
- [ ] 算子融合
- [ ] 批大小最大化

**通信优化**:
- [ ] NCCL backend
- [ ] 梯度压缩
- [ ] 梯度累积
- [ ] 计算通信重叠

**内存优化**:
- [ ] gradient_as_bucket_view
- [ ] 及时释放中间变量
- [ ] 显存碎片整理

---

## 💡 实践指南

### 快速开始: DataParallel
```python
import torch.nn as nn

# 1. 定义模型
model = YourModel()

# 2. 包装为多GPU (1行!)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 3. 移到GPU
model = model.to('cuda')

# 4. 正常训练
for X, y in dataloader:
    X, y = X.to('cuda'), y.to('cuda')
    output = model(X)  # 自动分割!
    loss = criterion(output, y)
    loss.backward()  # 自动聚合!
    optimizer.step()
```

### 进阶: DistributedDataParallel
```python
# train.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # 1. 初始化进程组
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # 2. 设置设备
    torch.cuda.set_device(local_rank)
    
    # 3. 创建模型
    model = YourModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # 4. 使用DistributedSampler
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler)
    
    # 5. 训练
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 重要!
        for X, y in dataloader:
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    main()
```

**启动命令**:
```bash
# 单机4卡
torchrun --nproc_per_node=4 train.py

# 2机,每机4卡
# 机器0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=192.168.1.1 --master_port=29500 train.py
# 机器1:  
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr=192.168.1.1 --master_port=29500 train.py
```

### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for X, y in dataloader:
    X, y = X.to('cuda'), y.to('cuda')
    
    optimizer.zero_grad()
    
    # 自动混合精度
    with autocast():
        output = model(X)
        loss = criterion(output, y)
    
    # 梯度缩放
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 🔍 性能调优流程

### 1. 建立基线
```bash
# 单GPU训练
python train.py --gpus 1 --batch-size 64

# 记录:
# - 训练时间: __秒/epoch
# - GPU利用率: __%
# - 显存使用: __GB
```

### 2. 扩展到多GPU
```bash
# 4 GPU训练
python train.py --gpus 4 --batch-size 256  # 64×4

# 期望:
# - 训练时间: ~基线/4
# - GPU利用率: >85%
# - 测试精度: 与单GPU相近
```

### 3. 识别瓶颈
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True) as prof:
    for _ in range(10):
        output = model(input)
        loss = criterion(output, target)
        loss.backward()

# 分析
print(prof.key_averages().table(sort_by="cuda_time_total"))

# 关注:
# - DataLoader时间 (应 < 5%)
# - 通信时间 (应 < 20%)  
# - 计算时间 (应 > 75%)
```

### 4. 针对性优化

**瓶颈1: 数据加载慢**
```python
# 解决方案
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=16,      # 增加worker
    pin_memory=True,     # 锁页内存
    prefetch_factor=4    # 预取
)
```

**瓶颈2: 通信开销大**
```python
# 解决方案1: 梯度累积
accumulation_steps = 4  # 减少通信频率

# 解决方案2: 梯度压缩
model = DDP(model, bucket_cap_mb=25)

# 解决方案3: 混合精度
使用FP16减少通信量
```

**瓶颈3: 显存不足**
```python
# 解决方案1: 梯度检查点
from torch.utils.checkpoint import checkpoint
output = checkpoint(model.layer, input)

# 解决方案2: 混合精度
节省50%显存

# 解决方案3: 模型并行
大模型拆分到多GPU
```

---

## 📊 性能基准

### 单机多GPU加速比 (理想 vs 实际)

| GPU数 | 理论加速 | 实际加速 | 效率   | 备注        |
|------|------|------|------|-----------|
| 1    | 1.0x | 1.0x | 100% | 基线        |
| 2    | 2.0x | 1.8x | 90%  | 通信开销~10%  |
| 4    | 4.0x | 3.4x | 85%  | PCIe带宽限制  |
| 8    | 8.0x | 6.4x | 80%  | 需NVLink优化 |

**影响因素**:
- 模型大小 (小模型通信占比高)
- 批大小 (太小GPU不饱和)
- 网络拓扑 (NVLink > PCIe)
- 框架开销 (Python GIL)

### 通信时间占比 (目标 < 20%)

```
计算密集型模型 (ResNet-50):
  计算: 80%
  通信: 15%
  其他: 5%

通信密集型模型 (BERT):
  计算: 60%
  通信: 35%
  其他: 5%

优化目标:
  - 增大批量 → 减少通信频率
  - 混合精度 → 减少通信量
  - All-Reduce → 高效聚合
```

---

## 🛠️ 常用工具

### 监控工具
```bash
# GPU状态
nvidia-smi

# 实时监控
watch -n 1 nvidia-smi

# 详细信息
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1

# 多GPU脚本
for i in {0..7}; do
  echo "GPU $i:"
  nvidia-smi -i $i --query-gpu=utilization.gpu,memory.used --format=csv,noheader
done
```

### Profiling工具
```python
# PyTorch Profiler
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(input)

# TensorBoard可视化
prof.export_chrome_trace("trace.json")

# 或使用
from torch.profiler import tensorboard_trace_handler
prof = profile(
    on_trace_ready=tensorboard_trace_handler('./log')
)
```

```bash
# NVIDIA工具
nvprof python train.py
nsight-systems profile python train.py
```

### 调试工具
```python
# 检测异常
torch.autograd.set_detect_anomaly(True)

# 同步检查
torch.cuda.synchronize()
# 然后检查错误

# 确定性运行
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)
```

---

## 📖 推荐资源

### 官方文档
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### 工具与库
- **Horovod**: Uber的分布式训练框架
- **DeepSpeed**: 微软的大模型训练优化
- **Megatron-LM**: NVIDIA的模型并行
- **PyTorch Lightning**: 简化分布式代码
- **Accelerate**: HuggingFace的训练加速库

### 论文
- Accurate, Large Minibatch SGD (FAIR, 2017)
- Ring All-Reduce (Baidu, 2017)  
- Parameter Server (CMU, 2014)
- ZeRO: Memory Optimizations (Microsoft, 2019)

---

## ✅ 学习检查

完成本章后,你应该能够:

**基础**:
- [ ] 理解CPU/GPU硬件架构差异
- [ ] 解释内存层次和缓存作用
- [ ] 掌握异步计算的工作原理
- [ ] 使用non_blocking异步传输

**进阶**:
- [ ] 实现数据并行的核心组件
- [ ] 使用nn.DataParallel训练模型
- [ ] 配置合适的批大小和学习率
- [ ] 监控GPU利用率和性能

**高级**:
- [ ] 部署DistributedDataParallel
- [ ] 实现混合精度训练
- [ ] 使用Profiler分析瓶颈
- [ ] 优化通信和计算重叠
- [ ] 理解参数服务器架构

**专家**:
- [ ] 从零实现Ring All-Reduce
- [ ] 设计自定义同步策略
- [ ] 处理多机训练故障
- [ ] 实现模型并行(大模型)

---

## 🎓 实战项目建议

1. **基础项目**: ResNet-50 单机4卡训练
   - 目标: 实现>3x加速
   - 难度: ⭐⭐
   - 时间: 1天

2. **进阶项目**: BERT-Base 混合精度训练
   - 目标: 省50%显存,快1.5x
   - 难度: ⭐⭐⭐
   - 时间: 2-3天

3. **高级项目**: GPT-2 多机多卡训练
   - 目标: 8机64卡,线性扩展
   - 难度: ⭐⭐⭐⭐
   - 时间: 1周

4. **专家项目**: 实现自定义通信Backend
   - 目标: 优化特定场景性能
   - 难度: ⭐⭐⭐⭐⭐
   - 时间: 2-4周

---

## 💬 常见问题 (FAQ)

**Q1: DataParallel vs DistributedDataParallel?**

A: 
- **DataParallel**: 单进程,简单但有GIL,GPU 0负载重
- **DDP**: 多进程,无GIL,负载均衡,推荐使用

**Q2: 批大小应该如何调整?**

A: 
```
经验法则: batch_size = base_batch × GPU数
例外情况:
  - BatchNorm: 可能需更大per-GPU batch
  - 小数据集: 总batch不要超过数据集大小
  - 显存限制: 用梯度累积模拟大batch
```

**Q3: 学习率要调整吗?**

A:
```
常用策略:
1. 线性缩放: lr = base_lr × GPU数
2. 平方根缩放: lr = base_lr × sqrt(GPU数)
3. Warmup: 前几个epoch线性增加lr

建议: 先试线性缩放+warmup
```

**Q4: GPU利用率低怎么办?**

A:
```
诊断:
1. nvidia-smi监控 → 找低利用率时段
2. Profiler分析 → 找耗时操作

常见原因:
- 数据加载慢 → 增加num_workers
- 批太小 → 增大batch_size
- 模型太小 → 考虑更大模型
- 同步太多 → 减少同步点
```

**Q5: OOM怎么解决?**

A:
```
优先级从高到低:
1. 混合精度 (FP16) → 省50%显存
2. 减小batch_size → 确保能运行
3. 梯度累积 → 模拟大batch
4. 梯度检查点 → 用时间换空间
5. 模型并行 → 最后手段
```

**Q6: 多GPU精度下降?**

A:
```
可能原因:
1. BatchNorm统计不准 → 用SyncBatchNorm
2. 批太大收敛差 → 调整学习率/warmup
3. 随机性不同 → 设置相同seed

验证方法:
  固定seed,单GPU vs 多GPU loss应相同
```

---

**下一章预告**: Chapter 12 - 计算机视觉 (图像分类、目标检测、语义分割)

---

**致谢**: 本章内容参考了PyTorch官方文档、NVIDIA深度学习最佳实践,以及业界最新研究成果。