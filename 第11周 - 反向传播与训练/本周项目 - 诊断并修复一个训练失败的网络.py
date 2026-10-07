import torch
from torch import nn, optim

torch.manual_seed(0)

# 随机生成训练数据，200个样本，每个样本20个特征，目标是一个二分类标签
X = torch.randn(200, 20)
y = (X.sum(dim=1) > 0).long()  # 构造一个简单的二分类任务


# 故意写一个"训练会失败"的版本:学习率过大 + 没有归一化
class BadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = BadNet()
optimizer = optim.SGD(model.parameters(), lr=10.0)  # 学习率故意设得过大
criterion = nn.CrossEntropyLoss()

print("=== 问题版本(学习率过大) ===")
for epoch in range(20):
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")  # loss会震荡甚至爆炸,不收敛


# 修复版本:合理学习率 + 加入BatchNorm + 用Adam
class GoodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)


model2 = GoodNet()
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

print("\n=== 修复版本 ===")
for epoch in range(20):
    optimizer2.zero_grad()
    out = model2(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer2.step()
    if epoch % 5 == 0:
        print(f"epoch {epoch}: loss={loss.item():.4f}")  # 应该平稳下降


# 跑完你会清楚看到两个版本loss曲线的巨大差异——这个对比本身就是本周所有技巧(学习率、BatchNorm、Adam)为什么重要的最直观证明。
# "训练不收敛"在实践中极其常见，诊断思路通常就是：
#       先查学习率是不是太大，再查有没有做归一化，再换成Adam试试。
