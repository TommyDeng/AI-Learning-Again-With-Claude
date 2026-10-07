import torch
from torch import nn


class NetWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.dropout = nn.Dropout(p=0.5)  # 训练时随机"关闭"50%的神经元
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = NetWithDropout()

# 关键:训练和评估模式下Dropout行为不同!
model.train()  # 训练模式:Dropout生效,随机丢弃神经元
dummy = torch.randn(4, 10)
print("训练模式输出(每次可能不同):\n", model(dummy))

model.eval()  # 评估模式:Dropout不生效,用全部神经元(这就是为什么之前代码里测试前要调用model.eval())
with torch.no_grad():
    print("评估模式输出:\n", model(dummy))

# 直觉理解: 训练时随机"屏蔽"一部分神经元，强迫网络不能过度依赖某几个特定神经元，相当于同时训练了很多个"瘦身版"子网络的集成——这能有效防止过拟合。
# 这也解释了为什么训练前要 model.train()，测试前要 model.eval()——漏掉这一步是PyTorch新手最常踩的坑之一。
