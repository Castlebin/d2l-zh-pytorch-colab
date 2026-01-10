# %% @title 5.5 模型保存和加载

# %% 5.5.1 加载和保存张量
import torch

x = torch.arange(4)
torch.save(x, 'x-file')
print("x=", x)

# 从文件中加载保存的张量
x_2 = torch.load('x-file')
print("x_2=", x_2)
print("x==x_2: ", x == x_2, "\n")

# %% 可以保存一个张量列表
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print(x2, "\n", y2)

# %% 可以保存张量字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)


# %% title 5.5.2 加载和保存模型参数
# 定义好的模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# %% 保存模型参数
torch.save(net.state_dict(), 'mlp.params')

# %% 从保存的文件中加载模型参数
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()

# 验证两个模型的计算结果
Y_clone = clone(X)
Y_clone == Y  # 完全一致



