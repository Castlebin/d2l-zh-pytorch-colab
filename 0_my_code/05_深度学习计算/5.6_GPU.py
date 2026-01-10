# %% @title 5.6 GPU

# %% @title 5.6.1 计算设备
"""
在 PyTorch 中，CPU 和 GPU可以用 torch.device('cpu') 和 torch.device('cuda')表示。
应该注意的是， cpu 设备意味着所有物理 CPU 和内存， 这意味着 PyTorch 的计算将尝试使用所有CPU核心。 然而，gpu设备只代表一个卡和相应的显存。 如果有多个GPU，我们使用torch.device(f'cuda:{i}') 来表示第 i 块GPU（从0开始）。 另外， cuda:0和 cuda是等价的。
"""
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')

# %% 查询可用的 GPU 数量 (cuda 设备)
torch.cuda.device_count()


# %% 获取设备
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')

    # Windows 环境 、AMD 设备、使用 torch_dml
    import platform
    if platform.system() == "Windows":
        try:
            import torch_directml
            if torch_directml.is_available() and torch_directml.device_count() >= i + 1:
                return torch_directml.device(i)
        except ImportError:
            pass

    # Apple M 芯片
    if torch.backends.mps.is_available() and i == 0:
        return torch.device("mps")

    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的 GPU，如果没有 GPU，则返回 [cpu(),]"""
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

    import platform
    if platform.system() == "Windows":
        try:
            import torch_directml
            if torch_directml.is_available():
                for i in range(torch_directml.device_count()):
                    devices.append(torch_directml.device(i))
        except ImportError:
            pass

    if torch.backends.mps.is_available():
        devices.append(torch.device('mps'))

    return devices if devices else [torch.device('cpu')]


try_gpu(), try_gpu(10), try_all_gpus()

# %% @title 5.5.2 张量与 GPU
"""默认情况下，张量是在CPU上创建的。"""
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x.device

# %% 将张量放在 GPU 上
X = torch.ones(2, 3, device=try_gpu())
X

# 不同设备上的张量无法一起计算
x_cpu = torch.tensor([1, 2, 3])
x_gpu = torch.tensor([1, 2, 3], device=try_gpu(0))

'''
x_cpu + x_gpu   # 无法一起计算，报错
'''

# %% 将 x_cpu 移动到 和 x_gpu 相同的设备上，就可以一起计算
x_cpu_2_gpu = x_cpu.to(try_gpu(0))

x_cpu_2_gpu * x_cpu_2_gpu



# %% @title 5.6.3 神经网络与 GPU
# 将神经网络放在 GPU 上
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

# %% 当输入为GPU上的张量时，模型将在同一 GPU 上计算结果。
y = net(torch.ones(2, 3, device=try_gpu()))

y

# %% 可以看到模型的参数也是在同样的设备上
net[0].weight.data.device

