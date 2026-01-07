import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from matplotlib_cn import matplotlib_util
matplotlib_util.enable_chinese()

# pytorch 快速入门教程
# https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

#%%
# 1. 加载数据集
# 训练数据集
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 测试数据集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# %%
# 数据可视化，查看一下前几个样本

# 标签映射
labels_map = {
    0: "T恤 T-Shirt",
    1: "裤子 Trouser",
    2: "套衫 Pullover",
    3: "连衣裙 Dress",
    4: "外套 Coat",
    5: "凉鞋 Sandal",
    6: "衬衫 Shirt",
    7: "运动鞋 Sneaker",
    8: "包 Bag",
    9: "短靴 Ankle Boot",
}

# 可视化前 9 个样本
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()


# %%
# 2. 创建数据加载器，用于将数据集分成小批量，并打乱数据。用于训练
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 查看一下数据形状
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}") # Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
    print(f"Shape of y: {y.shape} {y.dtype}")    # Shape of y: torch.Size([64]) torch.int64
    break


# %%
# 3. 定义模型
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),   # 输入层到隐藏层
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # 隐藏层到输出层
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print("模型", model)



# %%
# 4. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)    # 随机梯度下降优化器，设置学习率为 0.001


# %%
# 5. 训练和测试函数

# 训练函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    # 设置模型为训练模式
    model.train()

    # 记录每个批次的训练损失和准确率
    train_losses, train_accs = [], []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        # 计算准确率
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()

        # 保存每个批次的训练损失和准确率
        train_losses.append(loss)
        train_accs.append(correct / len(y))


    # 计算整个训练集的平均损失和准确率
    avg_loss = sum(train_losses) / len(train_losses)
    avg_acc = sum(train_accs) / len(train_accs)

    print(f"Train Error: \n Avg loss: {avg_loss:>8f}, Avg accuracy: {(100*avg_acc):>0.1f}% \n")

    # 返回训练集的平均损失和准确率
    return avg_loss, avg_acc


# 测试函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # 设置模型为评估模式
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f}, Accuracy: {(100*correct):>0.1f}%\n")

    # 返回测试集的平均损失和准确率
    return test_loss, correct


# %%
# 6. 运行训练和测试循环

# 训练周期
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")



# %%
# 7. 保存训练好的模型
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")



# %%
# 8. 加载训练好的模型
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))


# %%
# 9. 使用模型进行预测

classes = [
    "T恤  T-shirt/top",
    "裤子 Trouser",
    "套衫 Pullover",
    "连衣裙 Dress",
    "外套 Coat",
    "凉鞋 Sandal",
    "衬衫 Shirt",
    "运动鞋 Sneaker",
    "包 Bag",
    "短靴 Ankle boot",
]


# 设置模型为评估模式
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')




