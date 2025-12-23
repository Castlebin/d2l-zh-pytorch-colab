## 3.2. 线性回归从零开始实现

import random
import torch
import matplotlib.pyplot as plt
import numpy as np

from matplotlib_cn import matplotlib_chinese
matplotlib_chinese.enable_matplotlib_chinese()

### 3.2.1. 生成数据集
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


# %% # 画出生成的数据集的前两个特征和标签的散点图
def plot_data(features, labels):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(features[:, 0].numpy(), labels.numpy(), c='blue', s=1)
    plt.xlabel('Feature 1')
    plt.ylabel('Label')
    plt.title('Feature 1 vs Label')

    plt.subplot(1, 2, 2)
    plt.scatter(features[:, 1].numpy(), labels.numpy(), c='red', s=1)
    plt.xlabel('Feature 2')
    plt.ylabel('Label')
    plt.title('Feature 2 vs Label')

    plt.tight_layout()
    plt.show()

plot_data(features, labels)   # 可以看到标签 y 与特征 1 呈正相关，与特征 2 呈负相关，符合我们设定的线性关系。


### 3.2.2. 读取数据集
'''
训练模型时要对数据集进行遍历，每次抽取一小批量样本，
所以定义一个函数， 该函数能打乱数据集中的样本并以小批量方式获取数据。
'''
# %%
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 打乱样本顺序

    # 按批量大小取出小批量样本
    for i in range(0, num_examples, batch_size):
        # 样本的下标
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# %%
# 查看一下第一批数据的前 3 个样本
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print('X: ', X[:3], '\n y: ',  y[:3])
    break


### 3.2.3. 初始化模型参数

# %%
# 初始化权重和偏差
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

print('初始权重 w:', w.T)
print('初始偏置 b:', b)

### 3.2.4. 定义模型
# %%
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

### 3.2.5. 定义损失函数
# %%
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

### 3.2.6. 定义优化算法
# %%
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():  # 禁用梯度跟踪
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # 梯度清零


### 3.2.7. 训练
'''
在进行模型训练前，我们先会定义好一些超参数，例如学习率、批量大小和迭代次数等。

在每一次训练迭代中，我们需要遍历数据集一次（即一个epoch）。每次遍历需要做以下几步：
1. 取出一个小批量的数据 （batch）；
2. 通过模型计算预测值；
3. 计算损失函数值；
4. 反向传播计算梯度；
5. 使用优化算法更新模型参数。
'''
# %%
# 模型的超参数
lr = 0.03  # 学习率
num_epochs = 3  # 迭代次数
batch_size = 10  # 批量大小

# 使用线性回归模型和均方损失函数
net = linreg  # 线性回归模型
loss = squared_loss  # 均方损失函数

# 记录损失用于绘图
losses = []

# 开始执行训练过程
# 迭代 num_epochs 次。每个 epoch 都会遍历数据集一次
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0

    # 每次取出一个小批量数据进行训练
    for X, y in data_iter(batch_size, features, labels):
        # 计算预测值 (前向传播)
        y_hat = net(X, w, b)

        # 计算损失
        l = loss(y_hat, y)

        # 反向传播计算梯度
        l.sum().backward()

        # 使用小批量随机梯度下降更新参数
        sgd([w, b], lr, batch_size)

        epoch_loss += l.sum().item()
        num_batches += 1

    # 计算并记录当前 epoch 的平均损失
    avg_loss = epoch_loss / num_batches
    print(f'epoch {epoch + 1}, loss {avg_loss:.6f}')

    losses.append(avg_loss) # 记录损失用于绘图


#%% 可视化训练过程中的损失变化
plt.figure(figsize=(6, 4))
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('平均损失')
plt.title('训练损失曲线')
plt.grid(True, alpha=0.3)
plt.show()


# %%
# 查看一下训练好的模型参数与真实参数的差距
print(f'真实的 w: {true_w}, 估计的 w: {w.reshape(true_w.shape)}')
print(f'真实的 b: {true_b}, 估计的 b: {b}')

# 计算并打印估计误差。可以看到误差非常小，说明我们成功地学到了真实的线性模型参数
print(f'w 的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b 的估计误差: {true_b - b}')

