# %% @title 5.1 层和块
import torch
import torchinfo
from torch import nn

net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 查看模型
print("\n model: ", net)

X = torch.rand(2, 20)
net(X)

# %% 查看模型参数量
total_params = sum(p.numel() for p in net.parameters())  # 模型总参数量
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)  # 可训练参数量
non_trainable_params = total_params - trainable_params  # 不可训练参数量


# ========== 格式化输出（更易读，带单位） ==========
def format_num(num):
    """格式化数字，添加 K/M 单位"""
    if num >= 1e6:
        return f"{num / 1e6:.2f} M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f} K"
    else:
        return str(num)


print(f"模型总参数量: {format_num(total_params)} ({total_params:,})")
print(f"可训练参数量: {format_num(trainable_params)} ({trainable_params:,})")
print(f"不可训练参数量: {format_num(non_trainable_params)} ({non_trainable_params:,})")

# %% 直接使用 torchinfo 获取模型参数详细信息
torchinfo.summary(net)

# %% 5.1.1 自定义块
from torch.nn import functional as F


class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))


net_mlp = MLP()
print("\n net_mlp: ", net_mlp)

net_mlp(X)


# %% 5.1.2 顺序块
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module 是 Module 子类的一个实例。我们把它保存在成员变量 _modules 中，_modules 的类型是 OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)

        return X


net_my_seq = MySequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print("\n net_my_seq: ", net_my_seq)

net_my_seq(X)


# %% 5.1.3. 在前向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及 relu 和 mm 函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


net_fixed_hm = FixedHiddenMLP()
print("\n net_fixed_hm: ", net_fixed_hm)
net_fixed_hm(X)


# %% 我们可以混合搭配各种组合块的方法。 在下面的例子中，我们以一些想到的方法嵌套块。
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


# 通过混合搭配各种模型块，搭建自己的模型架构
chimera = nn.Sequential(
    NestMLP(),
    nn.Linear(16, 20),
    FixedHiddenMLP()
)
print(" \n chimera: ", chimera)
chimera(X)
