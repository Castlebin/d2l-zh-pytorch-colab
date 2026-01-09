# @title 4.4 模型选择、欠拟合和过拟合

# %% 导入必要的库
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
from dl_d2l import d2l_torch as d2l
from matplotlib_cn import matplotlib_util

matplotlib_util.enable_chinese()

# %% 生成数据: y = 5 + 1.2x - 3.4x² / 2! + 5.6x³ / 3! + noise
max_degree = 20                     # 多项式的最大阶数
n_train, n_test = 100, 100          # 训练集和测试集的大小
true_w = torch.zeros(max_degree)       # 分配参数向量
true_w[0:4] = torch.tensor([5, 1.2, -3.4, 5.6])  # 设置前四项的参数

# 生成特征
torch.manual_seed(42)
features = torch.randn((n_train + n_test, 1))
poly_features = torch.pow(features, torch.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= torch.tensor(math.factorial(i)) # Changed np.math.factorial to math.factorial

# 生成标签
labels = torch.matmul(poly_features, true_w)
labels += torch.normal(0, 0.1, size=labels.shape)

# 转换为float32
true_w, features, poly_features, labels = [x.float() for x in
                                            [true_w, features, poly_features, labels]]

print(f'特征形状: {features.shape}')
print(f'多项式特征形状: {poly_features.shape}')
print(f'标签形状: {labels.shape}')

# %% 可视化数据
plt.figure(figsize=(8, 5))
plt.scatter(features[:n_train].numpy(), labels[:n_train].numpy(),
            alpha=0.5, s=20, label='训练数据')
plt.scatter(features[n_train:].numpy(), labels[n_train:].numpy(),
            alpha=0.5, s=20, label='测试数据', marker='s')
plt.xlabel('x')
plt.ylabel('y')
plt.title('生成的数据')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# %% 定义评估函数
def evaluate_loss(net, data_iter, loss):  
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# %% 训练函数
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())

# %% 4.4.4.3. 三阶多项式函数拟合(正常)
# 我们将首先使用三阶多项式函数，它与数据生成函数的阶数相同。 结果表明，该模型能有效降低训练损失和测试损失。 学习到的模型参数也接近真实值

# 从多项式特征中选择前 4 个维度，即 1, x, x^2/2!, x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])


# %% 4.4.4.4. 线性函数拟合(欠拟合)
"""
让我们再看看线性函数拟合，减少该模型的训练损失相对困难。 在最后一个迭代周期完成后，训练损失仍然很高。 
当用来拟合非线性模式（如这里的三阶多项式函数）时，线性模型容易欠拟合。
"""
# 从多项式特征中选择前 2 个维度，即 1 和 x
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])



# %% 4.4.4.5. 高阶多项式函数拟合(过拟合)
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)


# %%
