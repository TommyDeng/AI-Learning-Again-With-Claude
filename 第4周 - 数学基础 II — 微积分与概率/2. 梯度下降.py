# 目标: 找到 f(x) = x^2 的最小值(答案显然是x=0)
def f(x):
    return x**2


def gradient(x):
    return 2 * x  # f(x)=x^2 的导数是 2x


x = 10.0  # 随便选个起点
learning_rate = 0.1
steps = 50

for i in range(steps):
    grad = gradient(x)
    x = x - learning_rate * grad  # 核心公式:沿梯度反方向移动
    if i % 10 == 0:
        print(f"第{i}步: x = {x:.4f}, f(x) = {f(x):.4f}")

print("最终结果:", x)  # 应该非常接近0


# 这就是所有神经网络训练的核心逻辑：不断计算梯度，朝梯度反方向小步移动，直到损失函数最小。
# 后面PyTorch的loss.backward()和optimizer.step()本质上就是自动帮你做这件事。
