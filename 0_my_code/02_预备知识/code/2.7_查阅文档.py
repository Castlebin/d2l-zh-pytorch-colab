# 2.7. 查阅文档

### 2.7.1. 查找模块中的所有函数和类

# %%
import torch

# dir() 函数可以列出模块中的所有函数和类
print(dir(torch.distributions))


### 2.7.2. 查找特定函数和类的用法

# %% help() 函数可以查看特定函数和类的用法
print(help(torch.ones))


