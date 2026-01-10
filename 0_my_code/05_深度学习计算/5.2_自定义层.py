# %% @title 5.4 自定义层

# %% @title 5.4.1 不带参数的层
import torch
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    # 输入参数，减去其均值
    def forward(self, X):
        return X - X.mean()


# 验证一下功能没问题
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

# %% 将层作为组件来构建模型
net = nn.Sequential(
    nn.Linear(8, 128),
    CenteredLayer()
)
X = torch.rand(4, 8)
Y = net(X)
print(Y.mean())  # 由于浮点数存在精度问题，所以这里输出不是 0 ，而是一个很小的接近 0 的数

# %% @title 5.4.2 带参数的层
# 实现一个自己的全连接层
from torch.nn import functional as F


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


# 试一下
linear = MyLinear(5, 3)
print("linear.weight: ", linear.weight, "\n")

Y_linear = linear(torch.randn(2, 5))
print("Y_linear: ", Y_linear, "\n")


# %% 使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层。
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)), "\n")


