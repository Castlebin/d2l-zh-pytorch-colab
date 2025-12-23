## 3.5. 图像分类数据集

'''
** Fashion-MNIST **是一个服饰图像数据集:
- **10个类别**: T恤、裤子、套衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴
- **训练集**: 60,000张图像
- **测试集**: 10,000张图像
- **图像尺寸**: 28×28像素,灰度图(单通道)

相比MNIST手写数字数据集,Fashion-MNIST更具挑战性,更接近实际应用。
'''

### 下载和加载数据集
# %%
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt


# ToTensor将图像转换为张量,并归一化到[0,1]
trans = transforms.ToTensor()

# 下载训练集和测试集
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

print(f'训练集大小: {len(mnist_train)}')
print(f'测试集大小: {len(mnist_test)}')
print(f'图像形状: {mnist_train[0][0].shape}')


# %% 可视化数据
from matplotlib_cn import matplotlib_chinese

matplotlib_chinese.enable_matplotlib_chinese()

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['T恤 T-shirt', '裤子 Trouser', '套衫 Pullover', '连衣裙 dress', '外套 coat',
                   '凉鞋 sandal', '衬衫 shirt', '运动鞋 sneaker', '包 bag', '短靴 ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.squeeze().numpy(), cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i], fontproperties='SimHei')
    plt.tight_layout()
    return axes

# 显示前18个样本
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y))
plt.show()


# 创建数据迭代器
batch_size = 256

# 创建数据加载器
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4)
test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4)

print(f'每个批次的形状:')
for X, y in train_iter:
    print(f'  X: {X.shape}, dtype: {X.dtype}')
    print(f'  y: {y.shape}, dtype: {y.dtype}')
    break

