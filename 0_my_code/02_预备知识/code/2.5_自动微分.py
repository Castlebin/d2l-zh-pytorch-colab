## 2.5. 自动微分

### 2.5.1. 一个简单的例子

# %%
import torch

x = torch.arange(4.0)
x


# %% 梯度
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None，因为还没有计算

# %% 定义一个新的张量y，它是x的函数
y = 2 * torch.dot(x, x)
y

# %% 调用反向传播函数来自动计算 y 关于 x 每个分量的梯度
y.backward()
x.grad   # tensor([ 0.,  4.,  8., 12.])

# %% 验证梯度是否正确
x.grad == 4 * x       # tensor([True, True, True, True])

# %% 现在计算x的另一个函数
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad              # tensor([1., 1., 1., 1.])


### 2.5.2. 非标量变量的反向传播

# %%
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x

# 等价于y.backward(torch.ones(len(x)))
y.sum().backward() # 对向量求和后再反向传播 （当输出不是标量时,需要先求和再反向传播）
x.grad             # tensor([0., 2., 4., 6.])


### 2.5.3. 分离计算
"""有时我们希望将某些计算移出计算图,使其被视为常数。"""
# %%
x.grad.zero_()
y = x * x
u = y.detach()  # 分离计算。u 与 y 具有相同的值，但不与计算图连接，因此后续 u 就不会被 跟踪梯度计算了，可以视作常数了，不会被后续的反向传播所影响
z = u * x

z.sum().backward()
x.grad == u             # tensor([True, True, True, True])

#%% 由于记录了y的计算结果，我们可以随后在 y 上调用反向传播， 正常得到 y = x*x 关于的 x 的导数，即 2*x
x.grad.zero_()
y.sum().backward()      # 当输出不是标量时,需要先求和再反向传播
x.grad == 2 * x         # tensor([True, True, True, True])



### 2.5.4. Python控制流的梯度计算
# %%
'''
使用自动微分的一个好处是： 
即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），
我们仍然可以计算得到的变量的梯度。 

在下面的代码中，while循环的迭代次数和if语句的结果都取决于输入 a 的值。
'''
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# %% 计算梯度
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

print(a.grad)
a.grad == d / a            # tensor(True)


### 2.5.5. 小结
'''
深度学习框架可以自动计算导数：
我们首先将梯度附加到想要对其计算偏导数的变量上，然后记录目标值的计算，执行它的反向传播函数，并访问得到的梯度。
'''



### 2.5.6. 练习



# %% 5. 绘制 f(x) = sin(x) 及其导数的图像。不使用 f'(x) = cos(x)
import torch
import matplotlib.pyplot as plt
import numpy as np

x = torch.linspace(0, 3*np.pi, 128)
x.requires_grad_(True)
y = torch.sin(x)  # y = sin(x)

y.sum().backward()

plt.plot(x.detach(), y.detach(), label='y=sin(x)')
plt.plot(x.detach(), x.grad, label='dy/dx=cos(x)')  # dy/dx = cos(x)
plt.legend(loc='upper right')
plt.show()



