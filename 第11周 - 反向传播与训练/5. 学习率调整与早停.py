import torch
from torch import nn, optim

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 学习率调度器:每30个epoch,学习率乘0.1
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# 随机生成训练数据，100个样本，每个样本10个特征，目标是一个标量
X = torch.randn(100, 10)
y = torch.randn(100, 1)
criterion = nn.MSELoss()

best_loss = float("inf")
patience = 10  # 容忍验证集loss多少轮不下降
patience_counter = 0

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    scheduler.step()  # 更新学习率

    # 早停逻辑(实际应该用独立的验证集loss,这里简化演示)
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"第{epoch}轮触发早停,loss不再下降")
        break

    if epoch % 20 == 0:
        print(
            f"epoch {epoch}: loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.5f}"
        )

# 早停(Early Stopping)的意义:
#   训练太久，模型会在训练集上越来越好，但在验证集上的表现会先降后升（过拟合开始）。
#   早停就是监控验证集表现，一旦连续若干轮不再改善，就提前停止训练，自动把"最佳时刻"的模型保留下来，而不是训练到最后一轮。
