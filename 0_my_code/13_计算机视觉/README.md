# 第12章: 计算机视觉

本章介绍深度学习在计算机视觉中的核心技术,包括图像增广、迁移学习、目标检测和语义分割。

## 📚 内容概览

### [01_图像增广与迁移学习.ipynb](./01_图像增广与迁移学习.ipynb)
- **图像增广技术**
  - 几何变换: 翻转、裁剪、旋转
  - 颜色变换: 亮度、对比度、饱和度、色调
  - 组合策略与实践
- **迁移学习与微调**
  - 预训练模型的使用
  - 模型微调策略
  - 学习率差异化设置
- **实战案例**
  - CIFAR-10图像分类
  - 热狗识别任务

### [02_目标检测基础.ipynb](./02_目标检测基础.ipynb)
- **边界框 (Bounding Box)**
  - 两种表示格式及转换
  - 坐标系统与可视化
- **锚框 (Anchor Box)**
  - 生成原理与参数设计
  - 多尺度锚框策略
  - 高效组合方法
- **多尺度目标检测**
  - 特征金字塔概念
  - 感受野与目标尺度匹配
  - 锚框数量优化
- **目标检测数据集**
  - 香蕉检测数据集
  - 批处理与填充策略

### [03_目标检测与语义分割模型.ipynb](./03_目标检测与语义分割模型.ipynb)
- **SSD (Single Shot MultiBox Detector)**
  - 单阶段检测架构
  - 多尺度特征图预测
  - 类别与边界框预测层
- **R-CNN系列**
  - R-CNN: 选择性搜索 + CNN
  - Fast R-CNN: RoI池化
  - Faster R-CNN: RPN区域提议网络
  - Mask R-CNN: 实例分割扩展
- **语义分割**
  - 任务定义与数据集(Pascal VOC2012)
  - FCN (全卷积网络)
  - 转置卷积与上采样
  - 双线性插值初始化

### [04_Kaggle实战_CIFAR10图像分类.ipynb](./04_Kaggle实战_CIFAR10图像分类.ipynb) ⭐⭐⭐
- **Kaggle竞赛实战**
  - CIFAR-10数据集组织
  - 原始图像文件处理
  - 图像增广综合应用
- **模型训练策略**
  - ResNet架构应用
  - 训练与验证集划分
  - 超参数调优
- **结果提交**
  - 预测格式化
  - CSV文件生成
  - Kaggle平台提交

### [05_Kaggle实战_狗品种识别.ipynb](./05_Kaggle实战_狗品种识别.ipynb) ⭐⭐⭐
- **ImageNet数据集**
  - 120类狗品种分类
  - 不同尺寸图像处理
- **迁移学习实战**
  - 预训练ResNet微调
  - 学习率调度策略
  - 数据增广技巧
- **模型优化**
  - K折交叉验证
  - 模型集成
  - 测试时增广(TTA)

### [06_神经风格迁移实战.ipynb](./06_神经风格迁移实战.ipynb) ⭐⭐
- **算法原理**
  - Gatys 风格迁移框架
  - 内容/风格/全变分损失
  - Gram 矩阵的作用
- **PyTorch 实现**
  - 使用 VGG19 特征网络
  - 构建损失函数与优化流程
  - 结果可视化与保存
- **调参技巧**
  - 损失权重的选择
  - 迭代次数与优化器对比
  - 多阶段/高分辨率策略

## 🎯 学习路线

```
第1周: 图像增广与迁移学习
  ├─ Day 1-2: 图像增广方法
  ├─ Day 3-4: 迁移学习原理
  └─ Day 5-7: 实战项目

第2周: 目标检测基础
  ├─ Day 1-2: 边界框与锚框
  ├─ Day 3-4: 多尺度检测
  └─ Day 5-7: 数据集处理

第3周: 检测模型
  ├─ Day 1-3: SSD模型
  ├─ Day 4-6: R-CNN系列
  └─ Day 7: 模型对比

第4周: 语义分割
  ├─ Day 1-3: FCN模型
  ├─ Day 4-5: 转置卷积
  └─ Day 6-7: 项目实践

第5-6周: Kaggle竞赛实战
  ├─ Week 5: CIFAR-10图像分类
  │   ├─ Day 1-2: 数据准备
  │   ├─ Day 3-5: 模型训练
  │   └─ Day 6-7: 调优提交
  └─ Week 6: 狗品种识别
      ├─ Day 1-2: 数据分析
      ├─ Day 3-5: 迁移学习
      └─ Day 6-7: 模型集成

附加: 神经风格迁移
  ├─ Day 1: 理解损失函数与Gram矩阵
  ├─ Day 2: 复现Notebook结果
  └─ Day 3: 尝试不同风格/调参与多阶段策略
```

## 🔑 核心概念

### 图像增广
**目的**: 扩充数据集,提高泛化能力

**常用方法**:
- ✅ **翻转**: `RandomHorizontalFlip`, `RandomVerticalFlip`
- ✅ **裁剪**: `RandomResizedCrop(size, scale, ratio)`
- ✅ **颜色**: `ColorJitter(brightness, contrast, saturation, hue)`
- ✅ **组合**: `transforms.Compose([...])`

**最佳实践**:
```python
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

### 迁移学习
**流程**:
```
ImageNet预训练 → 复制特征层 → 替换输出层 → 微调训练
```

**学习率策略**:
```python
optimizer = torch.optim.SGD([
    {'params': feature_params, 'lr': 0.001},   # 特征层: 小lr
    {'params': output_params, 'lr': 0.01}      # 输出层: 大lr (10×)
], momentum=0.9)
```

### 目标检测
**核心组件**:
1. **边界框**: `(x1, y1, x2, y2)` ↔ `(cx, cy, w, h)`
2. **锚框**: 预定义候选框,减少搜索空间
3. **多尺度**: 不同层检测不同大小目标

**锚框生成**:
- 尺度 $s_1, ..., s_n$
- 宽高比 $r_1, ..., r_m$
- 每像素: $n+m-1$ 个锚框
- 尺寸: $w = hs\sqrt{r}$, $h = hs/\sqrt{r}$

### 模型对比

| 模型 | 类型 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| SSD | 单阶段 | 快 | 中 | 实时检测 |
| YOLO | 单阶段 | 极快 | 中 | 实时应用 |
| Faster R-CNN | 两阶段 | 中 | 高 | 精度优先 |
| Mask R-CNN | 两阶段 | 慢 | 高 | 实例分割 |
| FCN | 分割 | 中 | 中 | 语义分割 |

## 💻 代码示例

### 1. 图像增广
```python
from torchvision import transforms

# 定义增广
aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5)
])

# 应用增广
augmented_img = aug(img)
```

### 2. 模型微调
```python
# 加载预训练模型
model = torchvision.models.resnet18(pretrained=True)

# 替换输出层
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 差异化学习率
optimizer = torch.optim.SGD([
    {'params': model.layer4.parameters(), 'lr': 0.001},
    {'params': model.fc.parameters(), 'lr': 0.01}
], momentum=0.9)
```

### 3. 边界框转换
```python
def box_corner_to_center(boxes):
    """(x1,y1,x2,y2) → (cx,cy,w,h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), axis=-1)
```

### 4. 锚框生成
```python
def multibox_prior(data, sizes, ratios):
    """生成多尺度锚框"""
    h, w = data.shape[-2:]
    num_anchors = len(sizes) + len(ratios) - 1
    
    # 生成中心点
    center_h = (torch.arange(h) + 0.5) / h
    center_w = (torch.arange(w) + 0.5) / w
    
    # 生成锚框尺寸
    # ... (详见notebook)
    
    return anchors  # shape: (1, h*w*num_anchors, 4)
```

### 5. SSD预测层
```python
# 类别预测
cls_predictor = nn.Conv2d(in_channels, 
                          num_anchors * (num_classes + 1),
                          kernel_size=3, padding=1)

# 边界框预测
bbox_predictor = nn.Conv2d(in_channels, 
                           num_anchors * 4,
                           kernel_size=3, padding=1)
```

### 6. FCN模型
```python
class SimpleFCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 特征提取
        resnet = torchvision.models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # 1×1卷积调整通道
        self.conv_cls = nn.Conv2d(512, num_classes, 1)
        
        # 转置卷积上采样
        self.upsample = nn.ConvTranspose2d(
            num_classes, num_classes,
            kernel_size=64, stride=32, padding=16
        )
    
    def forward(self, x):
        x = self.features(x)      # 下采样
        x = self.conv_cls(x)      # 调整通道
        x = self.upsample(x)      # 上采样
        return x
```

## 📊 数据集

### CIFAR-10
- **规模**: 60,000张32×32彩色图像
- **类别**: 10类(飞机、汽车、鸟等)
- **用途**: 图像分类、增广实验

### Pascal VOC2012
- **规模**: 约3,000张图像
- **任务**: 目标检测、语义分割
- **类别**: 20个对象类 + 背景
- **标注**: 边界框、像素级掩码

### COCO
- **规模**: 330,000+ 图像
- **任务**: 检测、分割、关键点
- **类别**: 80类日常对象
- **特点**: 复杂场景、多目标

## 🛠️ 实用工具

### 数据增广库
```bash
# Albumentations (强大的增广库)
pip install albumentations

# imgaug
pip install imgaug
```

### 预训练模型
```python
# torchvision模型
import torchvision.models as models
resnet50 = models.resnet50(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

# timm (更多模型)
import timm
efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
```

### 可视化工具
```python
# 边界框可视化
from matplotlib.patches import Rectangle

def draw_bbox(ax, bbox, label, color='r'):
    rect = Rectangle((bbox[0], bbox[1]), 
                     bbox[2]-bbox[0], bbox[3]-bbox[1],
                     fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(bbox[0], bbox[1], label, 
            fontsize=12, bbox=dict(facecolor=color, alpha=0.5))
```

## 🎓 实践项目

### 初级项目
1. **宠物分类器**: 使用迁移学习识别猫狗品种
2. **数据增广实验**: 对比增广对CIFAR-10性能的影响
3. **边界框标注工具**: 手动标注少量图像

### 中级项目
1. **人脸检测**: 使用SSD或YOLO检测人脸
2. **车辆检测**: 在行车视频中检测车辆
3. **语义分割**: 在VOC2012上训练FCN

### 高级项目
1. **实例分割**: 使用Mask R-CNN分割不同对象
2. **全景分割**: 结合语义和实例分割
3. **自定义检测器**: 为特定领域设计检测模型

## 📖 扩展阅读

### 经典论文
- **图像增广**: 
  - AutoAugment (2019)
  - RandAugment (2020)
  
- **目标检测**:
  - R-CNN (2014)
  - Fast/Faster R-CNN (2015)
  - YOLO series (2016-2023)
  - SSD (2016)
  - RetinaNet (2017)
  - EfficientDet (2020)

- **语义分割**:
  - FCN (2015)
  - U-Net (2015)
  - DeepLab series (2015-2018)
  - PSPNet (2017)

### 在线资源
- [Papers with Code - Object Detection](https://paperswithcode.com/task/object-detection)
- [Detectron2](https://github.com/facebookresearch/detectron2) - Facebook的检测库
- [MMDetection](https://github.com/open-mmlab/mmdetection) - OpenMMLab检测工具箱
- [YOLOv5/v8](https://github.com/ultralytics/yolov5) - 实时检测

## ⚠️ 常见问题

### Q1: 图像增广是否总是有益?
**A**: 不一定。过度增广可能:
- 改变语义(如文字翻转)
- 引入不合理变换(如医学图像颜色变化)
- 增加训练时间

建议: 根据任务选择合适的增广。

### Q2: 迁移学习什么时候有效?
**A**: 
- ✅ **有效**: 源和目标任务相似(如ImageNet→其他自然图像)
- ❌ **效果有限**: 任务差异大(如自然图像→医学图像)
- 💡 **技巧**: 即使任务不同,网络结构也可复用

### Q3: SSD vs Faster R-CNN如何选择?
**A**:
- **SSD**: 速度优先(实时视频、移动设备)
- **Faster R-CNN**: 精度优先(准确率要求高)
- **YOLO**: 速度和精度平衡

### Q4: 小目标检测困难怎么办?
**A**: 
1. 使用更高分辨率输入
2. 多尺度训练和测试
3. 特征金字塔网络(FPN)
4. 数据增广(复制粘贴小目标)

### Q5: 转置卷积产生棋盘效应?
**A**: 
- **原因**: kernel_size不能被stride整除
- **解决**: 
  1. 使用resize + 卷积
  2. PixelShuffle
  3. 选择合适的kernel_size

### Q6: Kaggle竞赛如何提升排名?
**A**:
- **数据**: 更强的增广、TTA(测试时增广)
- **模型**: 集成多个模型、不同架构组合
- **训练**: K折交叉验证、更长的训练
- **技巧**: 伪标签、知识蒸馏、混合精度训练

## 🚀 性能优化

### 训练加速
```python
# 1. 混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 2. 多GPU训练
model = nn.DataParallel(model)

# 3. 数据加载优化
DataLoader(dataset, num_workers=4, pin_memory=True)
```

### 推理加速
```python
# 1. TorchScript
traced_model = torch.jit.trace(model, example_input)

# 2. ONNX导出
torch.onnx.export(model, example_input, "model.onnx")

# 3. TensorRT (NVIDIA)
# 需要安装torch2trt或TensorRT
```

## 📝 总结

**关键要点**:
1. ✅ 图像增广是提高性能的低成本方法
2. ✅ 迁移学习适用于小数据集
3. ✅ 目标检测需要理解锚框和多尺度
4. ✅ 单阶段快,两阶段准
5. ✅ 语义分割关键是恢复空间分辨率
6. ✅ Kaggle竞赛是检验技能的好方法

**学习建议**:
- 从简单任务开始(CIFAR-10分类)
- 理解每个组件的作用(锚框、RoI池化等)
- 动手实现简化版模型
- 使用预训练模型快速原型
- 参加Kaggle竞赛积累经验
- 在实际项目中应用

**下一步**:
- 学习注意力机制(Transformer用于CV)
- 了解视频理解(动作识别、视频分割)
- 探索3D视觉(点云、深度估计)
- 研究多模态学习(CLIP, DALL-E)

---

💡 **记住**: 计算机视觉是实践性很强的领域,多做项目,多调参数,多看数据!
