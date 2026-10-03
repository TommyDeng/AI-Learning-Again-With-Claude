import matplotlib.pyplot as plt
import torch
from torch import nn, optim


def train_model(optimizer_name, X, y, epochs=200):
    model = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
    criterion = nn.MSELoss()

    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == "SGD_momentum":
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


torch.manual_seed(0)
X = torch.linspace(-5, 5, 100).unsqueeze(1)
y = X**2 + torch.randn(100, 1) * 2  # 一个非线性任务:拟合 y=x^2

plt.figure(figsize=(8, 5))
for name in ["SGD", "SGD_momentum", "Adam"]:
    torch.manual_seed(0)  # 保证每个优化器从同样的初始参数开始,公平对比
    losses = train_model(name, X, y)
    plt.plot(losses, label=name)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("不同优化器的收敛速度对比")
plt.show()

# 直觉理解:

#       SGD — 最朴素的梯度下降，每步严格沿梯度方向走，容易在"峡谷状"损失曲面里震荡
#       SGD + momentum — 加入"惯性"，像小球滚下山坡会积累速度，能冲过小的局部坑洼，收敛更快更稳
#       Adam — 目前最常用的默认选择，给每个参数自适应调整学习率，通常收敛快且不太需要精调学习率

# 实际项目里，不确定用什么优化器时，Adam 几乎总是一个安全的默认起点。
