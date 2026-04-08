### 2.6.1 基本概率论

#%%
# 计算投掷骰子的点数概率，并绘制估计概率随实验次数增加的变化情况
#import d2l_torch as d2l
from dl_d2l import d2l_torch as d2l
import torch
from torch.distributions import multinomial

fair_probs = torch.ones([6]) / 6
counts = multinomial.Multinomial(10, fair_probs).sample((1000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()


# %%
