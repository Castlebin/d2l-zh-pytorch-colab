## 2.4 微积分


#%%
### 2.4.1 导数和微分
'''
导数是描述函数变化率的工具。对于函数 f(x)，其在点 x 处的导数定义为：
f'(x) = lim(h→0) [f(x + h) - f(x)] / h
'''


# %% 导数计算
# 定义函数 f(x) = 3x^2 - 4x
def f(x):
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    """计算数值导数"""
    return (f(x + h) - f(x)) / h


# h -> 0 ，观察数值导数的变化
h = 0.1
print("当x=1时,f'(x)的数值近似:")
for i in range(5):
    print(f'h={h:.5f}, 数值导数={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

print("\n理论值: f'(1) = 6*1 - 4 = 2")


# %% 绘制函数及其切线
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体以支持中文显示
# -------------------------- 设置中文字体 start --------------------------
# 可以替换为你系统中已有的中文字体
plt.rcParams['font.sans-serif'] = [
    # Windows 优先
    'SimHei', 'Microsoft YaHei',
    # macOS 优先
    'PingFang SC', 'Heiti TC',
    # Linux 优先
    'WenQuanYi Micro Hei', 'DejaVu Sans'
]
# 修复负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# -------------------------- 设置中文字体 end --------------------------

x = np.arange(0, 3, 0.1)
plt.figure(figsize=(6, 4))
# plt.plot(x, f(x), label='f(x) = 3x² - 4x')
# 使用 latex 语法显示数学公式，不然指数会显示为 □
plt.plot(x, f(x), label='$f(x) = 3x^2 - 4x$')
plt.plot(x, 2 * x - 3, linestyle='--', label='切线 y=2x-3 (x=1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.title('函数及其在 x=1 处的切线')
plt.show()



#%%
### 2.4.2 偏导数


#%%
### 2.4.3 梯度


#%%
### 2.4.4 链式法则


#%%
### 2.4.6 练习


