import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)  # 输入4个特征 → 8个神经元
        self.fc2 = nn.Linear(8, 3)  # 8个神经元 → 输出3个类别
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNet()
print(model)

# 测试前向传播:随便喂一个假数据看看形状对不对
dummy_input = torch.randn(1, 4)  # 1个样本,4个特征
print("输入:", dummy_input)
print("输入形状:", dummy_input.shape)
output = model(dummy_input)
print("输出:", output)
print("输出形状:", output.shape)
