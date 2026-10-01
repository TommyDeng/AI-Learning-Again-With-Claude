import numpy as np

np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# 初始化参数
w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 1000
n = len(X)

for epoch in range(epochs):
    y_pred = w * X + b
    error = y_pred - y

    # 计算梯度(损失函数是均方误差 MSE)
    dw = (2 / n) * np.sum(error * X)
    db = (2 / n) * np.sum(error)

    # 更新参数
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 200 == 0:
        loss = np.mean(error**2)
        print(f"epoch {epoch}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print(f"最终: w={w:.4f}, b={b:.4f}")  # 应该接近 w=2, b=1
