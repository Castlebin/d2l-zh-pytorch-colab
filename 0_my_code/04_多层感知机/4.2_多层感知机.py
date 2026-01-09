# @title 4.2 多层感知机从零开始实现
# %% 导入必要的库
import os
import time

from dl_d2l.util import colab_util

# %% 数据目录
base_data_dir = colab_util.get_base_data_dir()
print(f'base data dir: {base_data_dir}')

datasets_dir = os.path.join(base_data_dir, 'ML', 'Datasets')
os.makedirs(datasets_dir, exist_ok=True)
print(f'datasets dir: {datasets_dir}')

# %% matplotlib 绘图支持中文显示
from dl_d2l.util import matplotlib_util

matplotlib_util.enable_chinese()

# %% 加载 Fashion-MNIST 数据集
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

batch_size = 256

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root=datasets_dir, train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root=datasets_dir, train=False, transform=trans, download=True)

# Windows 系统上 DataLoader 不支持多进程加载，会出错。num_workers=0 表示使用 main 进程加载数据。这也是默认值
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0) 
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=0)  

print(f'训练集大小: {len(mnist_train)}')
print(f'测试集大小: {len(mnist_test)}')


# %% 1. 初始化模型参数
from torch import nn

"""
网络结构: 784 → 256 → 10
"""
num_inputs = 784  # 输入层: 784个特征(28×28)。Fashion-MNIST图像的高和宽均为 28 像素，是灰度图像，因此通道数为 1，28×28×1=784
num_outputs = 10  # 输出层: 10个类别
num_hiddens = 256  # 隐藏层: 256个隐藏单元

# 第一层: 输入 → 隐藏
W1 = torch.normal(0, 0.01, size=(num_inputs, num_hiddens), requires_grad=True)
b1 = torch.zeros(num_hiddens, requires_grad=True)

# 第二层: 隐藏 → 输出
W2 = torch.normal(0, 0.01, size=(num_hiddens, num_outputs), requires_grad=True)
b2 = torch.zeros(num_outputs, requires_grad=True)

params = [W1, b1, W2, b2]

print(f'参数总数: {sum(p.numel() for p in params):,}')
print(f'  W1: {W1.shape} → {W1.numel():,} 参数')
print(f'  b1: {b1.shape} → {b1.numel():,} 参数')
print(f'  W2: {W2.shape} → {W2.numel():,} 参数')
print(f'  b2: {b2.shape} → {b2.numel():,} 参数')


# %% 2. 定义激活函数
# 为了引入非线性，我们在隐藏层使用 ReLU 激活函数。ReLU 的定义是对输入的每个元素返回其与 0 的较大值。
# 这里为了对模型有更深入的理解，我们自己实现了 ReLU 函数，而不是直接使用 PyTorch 内置的 relu 函数。
def relu(X):
    """ReLU激活函数"""
    return torch.max(X, torch.zeros_like(X))

# 测试
test_x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print('输入:', test_x)
print('ReLU输出:', relu(test_x))


# %% 3. 定义模型
# 因为我们忽略了图像的空间结构，所以在这里我们需要将每张输入图像从二维矩阵展平为一个向量
def net(X):
    """两层MLP模型"""
    # 展平输入: (batch_size, 1, 28, 28) → (batch_size, 784)
    X = X.reshape((-1, num_inputs))

    # 第一层: 输入 → 隐藏
    H = relu(torch.matmul(X, W1) + b1)

    # 第二层: 隐藏 → 输出
    return torch.matmul(H, W2) + b2

# 测试模型
for X, y in train_iter:
    output = net(X)
    print(f'输入形状: {X.shape}')
    print(f'输出形状: {output.shape}')
    break


# %% 4. 定义损失函数
loss_fn = nn.CrossEntropyLoss(reduction='none') # 交叉熵损失函数


# %% 5. 训练模型

# 优化函数: 小批量随机梯度下降 （ 类似于 torch.optim.SGD() ）
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 计算准确数量
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 评估模型在指定数据集上的准确率
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的准确率"""
    metric = [0.0, 0.0]  # 正确预测数、总数
    with torch.no_grad():
        for X, y in data_iter:
            metric[0] += accuracy(net(X), y)
            metric[1] += y.numel()
    return metric[0] / metric[1]

# 训练函数
def train_epoch(net, train_iter, loss, updater):
    """训练一个迭代周期"""
    metric = [0.0, 0.0, 0.0]  # 训练损失总和、训练准确度总和、样本数

    for X, y in train_iter:
        # 前向传播
        y_hat = net(X)
        l = loss(y_hat, y)

        # 反向传播
        l.sum().backward()
        updater(params, lr, X.shape[0])

        metric[0] += float(l.sum())
        metric[1] += float(accuracy(y_hat, y))
        metric[2] += y.numel()

    return metric[0] / metric[2], metric[1] / metric[2]

# 开始训练
num_epochs = 10     # 训练轮数 number of epochs
lr = 0.1            # 学习率 learning rate

train_losses, train_accs, test_accs = [], [], []

start = time.time()
for epoch in range(num_epochs):
    train_metrics = train_epoch(net, train_iter, loss_fn, sgd)  # 使用 sgd 优化函数
    test_acc = evaluate_accuracy(net, test_iter)

    train_losses.append(train_metrics[0])
    train_accs.append(train_metrics[1])
    test_accs.append(test_acc)

    print(f'epoch {epoch + 1}, '
          f'loss {train_metrics[0]:.3f}, '
          f'train acc {train_metrics[1]:.3f}, '
          f'test acc {test_acc:.3f}')

print('训练时间: %.2f 秒' % (time.time() - start))

# %% 可视化训练过程
from matplotlib import pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(range(1, num_epochs + 1), train_losses, marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练损失')
ax1.grid(True, alpha=0.3)

# 准确率曲线
ax2.plot(range(1, num_epochs + 1), train_accs, marker='o', label='训练集')
ax2.plot(range(1, num_epochs + 1), test_accs, marker='s', label='测试集')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('准确率')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f'\n最终结果:')
print(f'训练准确率: {train_accs[-1]:.2%}')
print(f'测试准确率: {test_accs[-1]:.2%}')




# @title 4.3 多层感知机简洁实现（使用 pytorch）
# %% 1. 定义模型
net_concise = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net_concise.apply(init_weights)

print('模型结构:')
print(net_concise)
print(f'\n参数总数: {sum(p.numel() for p in net_concise.parameters()):,}')



# %% 2. 定义损失函数和优化器
loss_concise = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net_concise.parameters(), lr=0.1)

print('损失函数:', loss_concise)
print('优化器:', trainer)


# %% 3. 训练模型
def train_epoch_concise(net, train_iter, loss, updater):
    """训练一个迭代周期(简洁版)"""
    metric = [0.0, 0.0, 0.0]
    net.train()  # 设置为训练模式

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()

        metric[0] += float(l.sum())
        metric[1] += float(accuracy(y_hat, y))
        metric[2] += y.numel()

    return metric[0] / metric[2], metric[1] / metric[2]

# %% 开始训练
num_epochs = 10
train_losses_c, train_accs_c, test_accs_c = [], [], []

start = time.time()
for epoch in range(num_epochs):
    train_metrics = train_epoch_concise(net_concise, train_iter, loss_concise, trainer)
    test_acc = evaluate_accuracy(net_concise, test_iter)

    train_losses_c.append(train_metrics[0])
    train_accs_c.append(train_metrics[1])
    test_accs_c.append(test_acc)

    print(f'epoch {epoch + 1}, '
          f'loss {train_metrics[0]:.3f}, '
          f'train acc {train_metrics[1]:.3f}, '
          f'test acc {test_acc:.3f}')

print('训练时间: %.2f 秒' % (time.time() - start))


# %% 4. 两种实现结果对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失对比
ax1.plot(range(1, num_epochs + 1), train_losses, marker='o', label='从零实现')
ax1.plot(range(1, num_epochs + 1), train_losses_c, marker='s', label='简洁实现')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练损失对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 测试准确率对比
ax2.plot(range(1, num_epochs + 1), test_accs, marker='o', label='从零实现')
ax2.plot(range(1, num_epochs + 1), test_accs_c, marker='s', label='简洁实现')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('测试准确率对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


