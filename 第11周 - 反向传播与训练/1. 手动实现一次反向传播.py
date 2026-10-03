import torch

# 构造一个最简单的例子: y = w*x + b, 损失用均方误差
# 目标:通过梯度下降让模型拟合 y = 2x + 1

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])  # 对应 y = 2x + 1

# requires_grad=True 告诉PyTorch:这两个变量需要计算梯度
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

learning_rate = 0.01

for epoch in range(1000):
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()  # 自动计算 dloss/dw 和 dloss/db

    with torch.no_grad():  # 更新参数时不需要记录梯度
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 梯度默认会累加,所以每轮算完要手动清零
    w.grad.zero_()
    b.grad.zero_()

    if epoch % 20 == 0:
        print(
            f"epoch {epoch}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}"
        )

print(f"\n最终: w={w.item():.4f}, b={b.item():.4f}")  # 应该接近 w=2, b=1


# 这段代码做的事，和你用 nn.Module + optimizer 写的训练循环完全一样，只是把 optimizer.zero_grad() / optimizer.step() 这些封装打开了，手动写出来：

# loss.backward() — 自动微分引擎(autograd)顺着计算图反向传播，算出每个 requires_grad=True 的张量的梯度，存在 .grad 属性里
# w -= learning_rate * w.grad — 就是第4周你手写的梯度下降更新公式
# w.grad.zero_() — 等价于 optimizer.zero_grad() 在背后做的事

# 理解了这个，你就真正理解了PyTorch的optimizer.step()黑盒里到底在算什么。
