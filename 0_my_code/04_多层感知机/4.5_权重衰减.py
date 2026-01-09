# @title 4.5 权重衰减

# %% L2 正则化
"""
为了限制模型的复杂度，防止过拟合，可以在损失函数中添加权重的 L2 范数作为惩罚项。

L = loss + λ / 2 * ||w||^2
"""

# %% @title 4.5.1 高纬线性回归

# %% 生成测试数据
'''
y = 0.05 + 0.01 * x_1 + 0.01 * x_2 + ... + 0.01 * x_n + noise

为了使过拟合的效果更加明显，我们可以将问题的维数增加到 d=200， 并使用一个只包含20个样本的小训练集。
'''
import torch
from dl_d2l import d2l_torch as d2l

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05

train_data = d2l.synthetic_data(true_w, true_b, n_train)  # 训练数据集 X_train, y_train
train_iter = d2l.load_array(train_data, batch_size)

test_data = d2l.synthetic_data(true_w, true_b, n_test)  # 测试数据集 X_test, y_test
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# %%
print(f'训练数据: {train_data[0].shape}')
print(f'测试数据: {test_data[0].shape}')


# %% @title 4.5.2 从零开始实现

# %% 1. 定义一个函数，用于初始化模型参数
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# %% 2. 定义 L2 范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


# %% 3. 训练代码
def train_scratch(lambd):
    w, b = init_params()  # 初始化模型参数

    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss

    num_epochs, lr = 100, 0.003

    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])

    train_losses, test_losses = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了 L2 范数惩罚项，
            # 广播机制使 l2_penalty(w) 成为一个长度为 batch_size 的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)

        if (epoch + 1) % 5 == 0:
            train_loss = d2l.evaluate_loss(net, train_iter, loss)
            test_loss = d2l.evaluate_loss(net, test_iter, loss)
            print(f'epoch {epoch + 1}, train loss: {train_loss}, test loss: {test_loss}')

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            animator.add(epoch + 1, (train_loss, test_loss))

    print('w 的 L2 范数是：', torch.norm(w).item())

    return w, b, train_losses, test_losses


# %% 4. 不使用 L2 惩罚项直接训练 - 无权重衰减   ( λ = 0 )
w_no_wd, b_no_wd, train_l_no_wd, test_l_no_wd = train_scratch(lambd=0)
d2l.plt.show()

# %% 5. 使用 L2 惩罚项训练 - 使用权重衰减
w_wd, b_wd, train_l_wd, test_l_wd = train_scratch(lambd=3)
d2l.plt.show()

# %% 可视化对比
from matplotlib_cn import matplotlib_util
from matplotlib import pyplot as plt

matplotlib_util.enable_chinese()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs = range(0, 100, 5)

# 训练损失
ax1.plot(epochs, train_l_no_wd, marker='o', label='无权重衰减')
ax1.plot(epochs, train_l_wd, marker='s', label='有权重衰减')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Train Loss')
ax1.set_title('训练损失对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 测试损失
ax2.plot(epochs, test_l_no_wd, marker='o', label='无权重衰减')
ax2.plot(epochs, test_l_wd, marker='s', label='有权重衰减')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Test Loss')
ax2.set_title('测试损失对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print('观察: 权重衰减提高了测试性能(泛化能力)!')

# %% @title 简洁实现 ( pytorch )
"""
PyTorch 的优化器自带 weight_decay 参数，表示使用惩罚项

在下面的代码中，我们在实例化优化器时直接通过 weight_decay 指定 weight decay 超参数。
默认情况下，PyTorch 同时衰减权重和偏移。 这里我们只为 权重 w 设置了weight_decay，所以偏置参数 b 不会衰减。
"""
from torch import nn


def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))

    for param in net.parameters():
        param.data.normal_()

    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003

    # weight 使用衰减、bias 偏置参数 没有使用衰减
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}
    ], lr=lr)

    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])

    train_losses, test_losses = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()

        if (epoch + 1) % 5 == 0:
            train_loss = d2l.evaluate_loss(net, train_iter, loss)
            test_loss = d2l.evaluate_loss(net, test_iter, loss)
            print(f'epoch {epoch + 1}, train loss: {train_loss}, test loss: {test_loss}')

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            animator.add(epoch + 1, (train_loss, test_loss))

    print('w 的 L2 范数：', net[0].weight.norm().item())

    return net[0].weight, net[0].bias, train_losses, test_losses


# %% 1. 不使用 L2 惩罚项直接训练 - 无权重衰减   ( λ = 0 )
w_no_wd_2, b_no_wd_2, train_l_no_wd_2, test_l_no_wd_2 = train_concise(0)
d2l.plt.show()

# %% 2. 使用 L2 惩罚项训练 - 使用权重衰减
w_wd_2, b_wd_2, train_l_wd_2, test_l_wd_2 = train_concise(3)
d2l.plt.show()

# 绘图对比可以看到效果是一样的，省去

