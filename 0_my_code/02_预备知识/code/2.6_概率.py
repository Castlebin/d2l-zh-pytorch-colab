##2.6. 概率

""" 简单地说，机器学习就是做出预测。 """


### 2.6.1. 基本概率论

# %% 计算投掷骰子的点数概率
import torch
from torch.distributions import multinomial

fair_probs = torch.ones([6]) / 6         # 理论上每个点数的概率相等，都是 1/6
print("理论概率:", fair_probs)

# %% 投掷骰子
p1 = multinomial.Multinomial(1, fair_probs).sample()        # 投掷一次骰子
print("p1:", p1)

# %% 模拟多次投掷骰子
p2 = multinomial.Multinomial(10, fair_probs).sample()      # 投掷 10 次骰子
# 查看结果分布
print("p2:", p2)

p3 = multinomial.Multinomial(10000, fair_probs).sample()      # 投掷 10000 次骰子
# 查看结果分布
print("p3:", p3)

p4 = multinomial.Multinomial(1000000, fair_probs).sample()      # 投掷 1000000 次骰子
# 查看结果分布
print("p4:", p4)
