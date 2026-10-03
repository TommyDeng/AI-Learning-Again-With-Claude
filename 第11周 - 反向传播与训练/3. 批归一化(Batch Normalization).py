import torch
from torch import nn


class NetWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.bn1 = nn.BatchNorm1d(32)  # 对这一层的输出做归一化
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


model = NetWithBN()
dummy = torch.randn(8, 10)  # batch_size=8
print(model(dummy).shape)


# 为什么需要它: 深层网络训练时，每一层输入的分布会随着前面层参数的更新不断变化("内部协变量偏移")，这会让训练变得不稳定、变慢。
# BatchNorm 把每一层的输出重新拉回"均值0、方差1"附近，让训练更稳定，通常也能用更大的学习率，收敛更快。
