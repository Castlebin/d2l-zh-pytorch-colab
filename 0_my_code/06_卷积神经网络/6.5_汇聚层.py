# %% @title 6.5. 汇聚层   (Pooling)

# %% @title 6.5.1. 最大汇聚层和平均汇聚层

# %% Pooling
import torch
from torch import nn
from dl_d2l import d2l_torch as d2l


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()

    return Y


# %% 测试一下
X = torch.tensor([
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0]
])
pool2d(X, (2, 2))


# %%
pool2d(X, (2, 2), 'avg')



# @title 6.5.2. 填充和步幅
"""
与卷积层一样，汇聚层也可以改变输出形状
默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同
"""
# %%
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X

# %%
pool2d = nn.MaxPool2d(3)
pool2d(X)

# %% 填充和步幅可以手动设定
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)

# %%
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)


# %% @title 6.5.3. 多个通道
X = torch.cat((X, X + 1), 1)
X

# %%
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)

