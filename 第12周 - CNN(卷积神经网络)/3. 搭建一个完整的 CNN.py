import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 卷积层部分:提取特征

        # 输入1通道(灰度图)，输出16通道，卷积核尺寸3x3，padding=1保持尺寸不变
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)

        # 输入16通道，输出32通道，卷积核尺寸3x3，padding=1保持尺寸不变
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

        # 全连接层部分:分类
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 28x28图经过两次池化变成7x7
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # 卷积+池化反复堆叠
        # 图片尺寸经过两次池化过程的变化： 28x28 -> 14x14 -> 7x7
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7

        # 展平成一维向量,送进全连接层，目的是为了做分类
        # 整个过程就像是把装在 32 个 $7 \times 7$ 盒子里的积木，按顺序倒出来排成了一长排，数据原本的数值一模一样，没有任何信息在物理上被删减。
        x = x.view(x.size(0), -1)

        # 再经过两层全连接层，输出分类结果
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = SimpleCNN()
print(model)

# 验证输出形状
dummy = torch.randn(4, 1, 28, 28)  # 4张单通道的28x28的灰度图
print("输出形状:", model(dummy).shape)  # 应该是 [4, 10]

# CNN的典型结构: 卷积+池化反复堆叠(负责"看懂图像里有什么特征")，
# 最后接几层全连接(负责"根据这些特征做分类决策")。
# 层数越深，能识别的特征越抽象——浅层识别边缘/颜色，深层能识别"眼睛""轮子"这种复杂形状组合。
