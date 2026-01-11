# @title 6.2. 图像卷积

# @title 6.2.1. 互相关运算

# %% 计算卷积
import torch

def corr2d(X, K):
    x_h, x_w = X.shape  # 输入矩阵的 高、宽
    k_h, k_w = K.shape  # kernel 的 高、宽

    y_h, y_w = x_h - k_h + 1, x_w - k_w + 1 # 结果的 高、宽
    y = torch.zeros(y_h, y_w)
    for i in range(y_h):
        for j in range(y_w):
            y[i, j] = (X[i:i + k_h, j:j + k_w] * K).sum()

    return y


# %% 测试一下
X = torch.tensor([
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0]
])
K = torch.tensor([
    [0.0, 1.0],
    [2.0, 3.0]
])

corr2d(X, K)


# @title 6.2.2. 卷积层
# %% 定义卷积层
from torch import nn

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# @title 6.2.3. 图像中目标的边缘检测
# %% 构造一个 6*8 像素的黑白图像。中间四列为黑色（0），其余像素为白色（1）。
X = torch.ones((6, 8))
X[:, 2:6] = 0
X
"""
tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.],
        [1., 1., 0., 0., 0., 0., 1., 1.]])
"""

# %% 构造一个 1*2 的卷积核 K。当进行互相关运算时，如果水平相邻的两元素相同，则输出为零，否则输出为非零
K = torch.tensor([[1.0, -1.0]])

# %% 测试一下输出
Y = corr2d(X, K)
Y
"""
tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
        
 如所示，输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为
"""

# %% 将二维图像 X 转置一下，再计算
Y_t = corr2d(X.T, K)
Y_t
"""
tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])

从这个输出中可以看出，无法检测到边缘信息。所以，这个卷积核 K 只可以检测垂直边缘，无法检测水平边缘
"""

# %% @title 6.2.4. 学习卷积核
# 构造一个二维卷积层，它具有 1 个输出通道和形状为（1，2）的卷积核。为了方便，这里直接使用 torch 的卷积层，并且忽略偏置 bias
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

# 训练卷积核
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

# %% 查看当前学习到的卷积核参数
# 在 10 次迭代之后，误差已经降到足够低。现在我们来看看我们所学的卷积核的权重张量。
conv2d.weight  # 可以看到，非常接近我们之前定义的卷积核 [1.0, -1.0]

