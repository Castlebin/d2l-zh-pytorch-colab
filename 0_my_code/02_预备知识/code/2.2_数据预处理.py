## 2.2. 数据预处理

## 2.2.1. 读取数据集
# 在实际项目中,我们经常需要处理 CSV、Excel 等格式的原始数据。Pandas 是 Python 中最常用的数据分析工具。
# %%
import os

# 创建数据目录
# 创建数据目录
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')

# 写入 CSV 文件
with open(data_file, 'w', encoding='utf-8') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

print("CSV 文件已创建", )

# %% 使用 pandas 读取 CSV 文件
import pandas as pd

data = pd.read_csv(data_file)
print(data)

### 2.2.2. 处理缺失数据
'''
注意，“NaN”项代表缺失值。 为了处理缺失的数据，典型的方法包括插值法和删除法， 
其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。 在这里，我们将考虑插值法。

通过位置索引iloc，我们将data分成inputs和outputs， 其中前者为data的前两列，而后者为data的最后一列。 
对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
'''
#%%
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2:]

print('\ninputs:', inputs)
print('\noutputs:', outputs)

# 将数据分为输入和输出
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# 用均值填充数值列的缺失值
# inputs = inputs.fillna(inputs.mean())  # 报错，数据类型问题！fixed

# 用均值填充数值列的缺失值，并使用numeric_only=True确保只对数值列进行操作
# 非数值列的缺失值（如'Alley'列）将保留，后续由pd.get_dummies处理
inputs = inputs.fillna(inputs.mean(numeric_only=True))

print("填充后的inputs:")
print(inputs)

print(outputs)  # 为什么打印出的 outputs 没有表头这一行？

print('type(inputs):', type(inputs))
print('type(outputs):', type(outputs))
'''
打印出的 outputs 没有表头是因为 outputs 是一个 pandas.Series 对象，而不是一个 pandas.DataFrame 对象。

在 Pandas 中，当你通过 data.iloc[:, 2] 选择单列时，返回的是一个 Series 对象。Series 在打印时只显示索引和值，不会显示列名（表头）。
如果你希望打印时显示表头，可以将 Series 转换为 DataFrame。
```
outputs = outputs.to_frame()

print("输出数据 (带表头):")
print(outputs)
```

或者也还是用` data.iloc[:, 2:] `来选择最后一列，这样返回的就是 DataFrame，自然会显示表头。

'''

#### 2.2.2.1 处理离散值
'''对于类别特征,我们使用独热编码(one-hot encoding)将其转换为数值。'''
# 对于类别特征（离散值），我们可以使用独热编码（one-hot encoding）将每个类别值转换为一个新的二进制列。
# 在本示例中，Alley 列有两个可能的值：Pave 和 NaN。 pd.get_dummies() 可以为这一列创建两个新列，分别表示这两个值是否存在于原始数据中。
# %%
inputs = pd.get_dummies(inputs, dummy_na=True)
print("\n独热编码后的inputs:", inputs)

### 2.2.3. 转换为张量格式
# 最后，我们将数据转换为张量格式，以便在深度学习模型中使用。
# %%
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(f'X = {X} \n y = {y} ', X, y)

### 2.2.4. 小结
'''
pandas软件包是Python中常用的数据分析工具中，pandas可以与张量兼容。

用pandas处理缺失的数据时，我们可根据情况选择用插值法和删除法。
'''

### 2.2.5. 练习
'''
创建包含更多行和列的原始数据集。

删除缺失值最多的列。

将预处理后的数据集转换为张量格式。
'''
