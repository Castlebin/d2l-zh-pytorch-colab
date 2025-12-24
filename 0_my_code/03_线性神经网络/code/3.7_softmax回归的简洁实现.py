## 3.7. softmax回归的简洁实现

#%%
# 使用 Pytorch 实现 softmax 回归

import torch
from torch import nn
from torch.nn import functional as F

### 3.7.1. 初始化模型参数
# PyTorch不会隐式地调整输入的形状,所以需要flatten
# 定义模型
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)

# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

print('模型结构:')
print(net)

# %%
### 3.7.2. 重新审视Softmax的实现

# PyTorch的`CrossEntropyLoss`结合了Softmax和交叉熵,数值更稳定。
loss = nn.CrossEntropyLoss(reduction='none')
print('损失函数:', loss)


# %%
### 3.7.3. 优化算法
# 在这里，我们使用学习率为 0.1 的小批量随机梯度下降作为优化算法。
# 这与我们在线性回归例子中的相同，这说明了优化器的普适性。
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
print('优化器:', trainer)


# %%
### 3.7.4. 训练
def train_epoch(net, train_iter, loss, updater):
    """训练一个迭代周期"""
    metric = [0.0, 0.0, 0.0]  # 训练损失总和、训练准确度总和、样本数

    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)

        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器
            l.sum().backward()
            updater(X.shape[0])

        metric[0] += float(l.sum())
        metric[1] += float(accuracy(y_hat, y))
        metric[2] += y.numel()

    # 返回训练损失和训练准确率
    return metric[0] / metric[2], metric[1] / metric[2]

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的准确率"""
    metric = [0.0, 0.0]  # 正确预测数、总数
    with torch.no_grad():
        for X, y in data_iter:
            metric[0] += accuracy(net(X), y)
            metric[1] += y.numel()
    return metric[0] / metric[1]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

        train_losses.append(train_metrics[0])
        train_accs.append(train_metrics[1])
        test_accs.append(test_acc)

        print(f'epoch {epoch + 1}, '
              f'loss {train_metrics[0]:.3f}, '
              f'train acc {train_metrics[1]:.3f}, '
              f'test acc {test_acc:.3f}')

    return train_losses, train_accs, test_accs


num_epochs = 10
batch_size = 256

# 加载训练数据集、测试集
from dl_d2l import d2l_torch as d2l

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 开始训练
train_losses_c, train_accs_c, test_accs_c = train(
    net, train_iter, test_iter, loss, num_epochs, trainer)


# %%
# 可视化训练过程
import matplotlib.pyplot as plt
from dl_d2l.util import enable_matplotlib_chinese
enable_matplotlib_chinese()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失
ax1.plot(range(1, num_epochs + 1), train_losses_c, marker='s', label='简洁实现')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练损失')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 测试准确率
ax2.plot(range(1, num_epochs + 1), test_accs_c, marker='s', label='简洁实现')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('测试准确率')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%
# 预测、可视化
def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['T恤 T-shirt', '裤子 Trouser', '套衫 Pullover', '连衣裙 dress', '外套 coat',
                   '凉鞋 sandal', '衬衫 shirt', '运动鞋 sneaker', '包 bag', '短靴 ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.squeeze().numpy(), cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i], fontproperties='SimHei')
    plt.tight_layout()
    return axes

def predict(net, test_iter, n=6):
    """预测标签"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [f'真实: {true}\n预测: {pred}'
              for true, pred in zip(trues, preds)]
    show_images(X[0:n], 1, n, titles=titles[0:n])
    plt.show()

# 预测
predict(net, test_iter)

