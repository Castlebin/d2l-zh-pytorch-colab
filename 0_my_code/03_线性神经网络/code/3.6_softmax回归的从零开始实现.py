## 3.6. softmax回归的从零开始实现

import torch
from dl_d2l import d2l_torch as d2l
import matplotlib.pyplot as plt

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

# %%
### 3.6.1. 初始化模型参数
num_inputs = 784  # 28 * 28
num_outputs = 10  # 10个类别

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

print(f'权重W形状: {W.shape}')
print(f'偏置b形状: {b.shape}')


# %%
### 3.6.2. 定义 softmax 操作
def softmax(X):
    """Softmax函数"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 广播机制

# 测试softmax
X_test = torch.normal(0, 1, (2, 5))
X_prob = softmax(X_test)
print('输入:')
print(X_test)
print('\nSoftmax输出(概率):')
print(X_prob)
print('\n每行的和:', X_prob.sum(1))


# %%
### 3.6.3. 定义模型
def net(X):
    # y = WX + b
    Y = torch.matmul(X.reshape((-1, W.shape[0])), W) + b
    return softmax(Y)  # softmax 输出


# %%
### 3.6.4. 定义损失函数

def cross_entropy(y_hat, y):
    """交叉熵损失函数"""
    return -torch.log(y_hat[range(len(y_hat)), y])

# 测试
y_test = torch.tensor([0, 2])
y_hat_test = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print('交叉熵损失:', cross_entropy(y_hat_test, y_test))


# %%
### 3.6.5. 计算分类准确率

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 测试
print('准确预测数量:', accuracy(y_hat_test, y_test))
print('准确率:', accuracy(y_hat_test, y_test) / len(y_test))


# %% 评估模型在数据集上的准确率
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的准确率"""
    metric = [0.0, 0.0]  # 正确预测数、总数
    with torch.no_grad():
        for X, y in data_iter:
            metric[0] += accuracy(net(X), y)
            metric[1] += y.numel()
    return metric[0] / metric[1]


# %%
### 3.6.6. 训练

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

def updater(batch_size):
    """SGD优化器"""
    return torch.optim.SGD([W, b], lr=0.1)

# 或者手动实现SGD
lr = 0.1

def sgd_updater(batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in [W, b]:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


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

# %%
# 开始训练

num_epochs = 10
batch_size = 256

# 加载训练数据集、测试集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 训练
train_losses, train_accs, test_accs = train(
    net, train_iter, test_iter, cross_entropy, num_epochs, sgd_updater)


# %%
# 可视化训练过程
from matplotlib_cn import matplotlib_util
matplotlib_util.enable_chinese()

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


# %% 3.6.7. 预测
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

predict(net, test_iter)


