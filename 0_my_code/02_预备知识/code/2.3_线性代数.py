## 2.3 线性代数

### 2.3.1 标量
# %%
import torch

# 标量是只有一个元素的张量
x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y)
print(x - y)
print(x * y)
print(x / y)

print(x ** y)

### 2.3.2 向量
'''
向量是一维张量，可以表示为一个数值序列
大量文献认为 列向量 是向量的默认方向，这是数学中的惯例
'''

# %%
# 创建一个向量
x = torch.arange(4)
print('x: ', x)  # tensor([0, 1, 2, 3])
print('x.shape: ', x.shape)  # torch.Size([4])

# 从 python 数组创建一个向量
y = torch.tensor([0, 1, 2, 3])
print('y: ', y)  # tensor([0, 1, 2, 3])
print('y.shape: ', y.shape)  # torch.Size([4])

print('x.dtype:', x.dtype)  # torch.int64
print('y.dtype: ', y.dtype)  # torch.int64

print('x == y : ', x == y)  # tensor([True, True, True, True])

# 可以通过索引访问向量中的元素。向量中的元素是标量
print('x[3]: ', x[3])  # tensor(3)

#### 2.3.2.1 长度、维度和形状
# %%
print(len(x))  # 4

print(x.shape)  # torch.Size([4])

### 2.3.3 矩阵
# %%
# 矩阵是二维张量，可以表示为一个数值表格

# 创建一个 5 * 4 的矩阵
A = torch.arange(20).reshape(5, 4)
print('A: ', A)

# 矩阵的形状
print('A.shape: ', A.shape)  # torch.Size([5, 4])

# %% 通过索引来访问矩阵中的任一元素
print('A[1,2]: ', A[1, 2])  # tensor(6)

# 访问矩阵的某一行
print('A[1]: ', A[1])  # tensor([4, 5, 6, 7])

# 访问矩阵的某一列
print('A[:, 1]: ', A[:, 1])  # tensor([ 1,  5,  9, 13, 17])

# A.T 表示矩阵 A 的转置
print("A.T: ", A.T)

# 对称矩阵
# %% 有一种特殊的矩阵，称为对称矩阵，即矩阵等于其转置矩阵
B = torch.tensor([[1, 2, 3],
                  [2, 0, 4],
                  [3, 4, 5]])
print("B: ", B)
print("B.T: ", B.T)
print("B == B.T: ", B == B.T)
'''
B == B.T:  tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
'''

### 2.3.4 张量
# %%
# 张量（本小节中的“张量”指代数对象）是描述具有任意数量轴的 n 维数组的通用方法。
# 例如，向量是一阶张量，矩阵是二阶张量。

# 创建一个 2 * 3 * 4 的三维张量
X = torch.arange(24).reshape(2, 3, 4)
print('X: ', X)

### 2.3.5 张量计算的基本性质
# %%
# 张量之间的加法和减法要求它们的形状相同，这些 **按元素** 的操作，不会改变张量的形状
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B

print('B: ', B)

C = A + B
print('A + B: ', A + B)

print('A.shape: ', A.shape)  # torch.Size([5, 4])
print('B.shape: ', B.shape)  # torch.Size([5, 4])
print('C.shape: ', C.shape)  # torch.Size([5, 4])

### 2.3.6 降维操作
# 有一些操作，可以将张量的维数减少，例如求和操作

# %% 对张量的所有元素求和，结果是一个标量
a = torch.arange(20, dtype=torch.float32).reshape(5, 4)  # 一个 5 行 4 列的矩阵
print('a.sum(): ', a.sum())  # tensor(190.)

# %% 可以指定某一个数轴对张量进行求和来降低维度
print('a.sum(dim=0): ', a.sum(dim=0))  # 在第 0 维（行）上求和，所以第 0 维被消除。结果是一个长度为 4 的向量

print('a.sum(dim=1): ', a.sum(dim=1))  # 在第 1 维（列）上求和，所以第 1 维被消除。结果是一个长度为 5 的向量

# %% 沿着行和列对矩阵进行求和，等价于对矩阵的所有元素求和
print('a.sum(dim=(0, 1)): ', a.sum(dim=(0, 1)))  # tensor(190.)

#### 2.3.6.1 非降维求和
# %%
sum_A = A.sum(axis=1, keepdims=True)
print("sum_A: ", sum_A)
print('sum_A.shape: ', sum_A.shape)  # torch.Size([5, 1]) 仍是 2 维张量

# %% 广播机制
print('A / sum_A : ', A / sum_A)     # 可以看到进行了广播之后，再进行的按元素除法

# %% 沿着某个数轴计算矩阵的累计和  （不会降低张量的维度）
cum_A = A.cumsum(dim=0)
print('cum_A: ', cum_A)


### 2.3.7 点积（Dot Product）
# %% 点积是向量运算中的一种基本操作。给定两个长度相同的向量，其点积是对应元素乘积的和。结果是一个标量
y = torch.arange(4, dtype=torch.float32)
x = torch.ones(4, dtype=torch.float32)
print('x.dot(y): ', x.dot(y))  # tensor(6.)

