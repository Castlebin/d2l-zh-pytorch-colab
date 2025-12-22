## 3.3. 线性回归的简洁实现

'''
从 0 开始实现线性回归虽然能帮助我们理解其基本原理，但在实际应用中，我们通常会使用深度学习框架提供的高级接口来简化这一过程。
这些高级接口封装了许多底层细节，使得我们能够更专注于模型设计和训练过程。

我们使用 PyTorch 框架来实现线性回归的简洁版本
'''

# %%
import torch
from torch.utils import data

# 所有的步骤起始都和从 0 开始实现线性回归时一样。只是我们使用 PyTorch 提供的高级 API 来简化实现过程。

### 3.3.1. 生成数据集
# %%
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))  # 每一行是一个样本，每一列是一个特征
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声，使用均值为 0，标准差为 0.01 的 正态分布 噪声
    return X, y.reshape((-1, 1))

# 假设真实的线性模型参数如下：
# 即真实模型 y = 2 * x1 - 3.4 * x2 + 4.2
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 现在生成 1000 个样本的数据集
features, labels = synthetic_data(true_w, true_b, 1000)


# 可以看到后面的实现中，我们不需要再手动实现数据迭代器、模型参数初始化、损失函数和优化算法等步骤，PyTorch 都为我们提供了现成的模块。

### 3.3.2. 读取数据集
# %%
batch_size = 10
dataset = data.TensorDataset(features, labels)
data_iter = data.DataLoader(dataset, batch_size, shuffle=True)


### 3.3.3. 定义模型
# %%
# nn 是 Neural Network 的缩写，包含了构建和训练神经网络的各种模块和函数
from torch import nn

'''使用`nn.Sequential`构建模型,`nn.Linear`是全连接层。'''
net = nn.Sequential(nn.Linear(2, 1))  # 定义一个包含单个线性层的神经网络，输入维度为 2，输出维度为 1
print('模型结构：\n', net)


### 3.3.4. 初始化模型参数
# %%
# 访问第一层(索引 0)，然后访问权重和偏置。设置初始值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

print('初始化后的参数:')
print('权重:', net[0].weight.data)
print('偏置:', net[0].bias.data)

### 3.3.5. 定义损失函数
# %%
loss = nn.MSELoss()   # 直接使用 PyTorch 提供的均方误差损失函数
print('损失函数：', loss)


### 3.3.6. 定义优化算法
# %%
from torch import optim
trainer = optim.SGD(net.parameters(), lr=0.03)  # 使用随机梯度下降优化器


### 3.3.7. 训练模型
# %%
# 记录损失用于绘图
losses = []

num_epochs = 3  # 迭代次数
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0

    # 每次取出一个小批量数据进行训练
    for X, y in data_iter:
        y_hat = net(X) # 计算预测值 (前向传播)

        l = loss(y_hat, y)  # 计算损失

        trainer.zero_grad()  # 梯度清零

        l.backward()  # 反向传播计算梯度

        trainer.step()  # 更新参数

        epoch_loss += l.sum().item()
        num_batches += 1

    l = loss(net(features), labels)  # 计算在整个数据集上的损失
    print(f'epoch {epoch + 1}, loss {l:f}')

    # 计算并记录当前 epoch 的平均损失
    avg_loss = epoch_loss / num_batches
    print(f'epoch {epoch + 1}, loss {avg_loss:.6f}')

    losses.append(avg_loss)  # 记录损失用于绘图


#%% 设置中文字体以支持中文显示
# -------------------------- 设置中文字体 start --------------------------
# 可以替换为你系统中已有的中文字体
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = [
    # Windows 优先
    'SimHei', 'Microsoft YaHei',
    # macOS 优先
    'PingFang SC', 'Heiti TC',
    # Linux 优先
    'WenQuanYi Micro Hei', 'DejaVu Sans'
]
# 修复负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 设置中文字体 end --------------------------


#%% 可视化训练过程中的损失变化
plt.figure(figsize=(6, 4))
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('平均损失')
plt.title('训练损失曲线')
plt.grid(True, alpha=0.3)
plt.show()



w = net[0].weight.data
b = net[0].bias.data
print(f'真实的 w: {true_w}, 估计的 w: {w.reshape(true_w.shape)}')
print(f'真实的 b: {true_b}, 估计的 b: {b}')
print('w的估计误差：', true_w - w.reshape(true_w.shape))
print('b的估计误差：', true_b - b)

