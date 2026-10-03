import matplotlib.pyplot as plt
import torch

# 1. 生成自变量（未激活前的输入数据）
x = torch.linspace(-5, 5, 100)

# 2. 计算激活后的输出
relu = torch.relu(x)
sigmoid = torch.sigmoid(x)
tanh = torch.tanh(x)

# 为什么需要激活函数:
#       如果神经网络每一层都只是线性运算(矩阵乘法+加法)，无论叠多少层，整个网络等价于一个单层线性模型
#       ——激活函数引入非线性，才让网络能学习复杂的模式(比如图像、语言这种高度非线性的数据)。
#       ReLU是目前最常用的，因为计算简单、能缓解梯度消失问题。

# 3. 创建画布并统一设置坐标轴
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 统一的坐标轴显示范围
x_limits = (-5, 5)
y_limits = (-5, 5)

activations = [(relu, "ReLU"), (sigmoid, "Sigmoid"), (tanh, "Tanh")]

for ax, (y_act, title) in zip(axes, activations):
    # 绘制未被激活函数处理前的图像 (y = x)
    ax.plot(x, x, label="Linear (y = x)", color="gray", linestyle="--", alpha=0.7)

    # 绘制激活函数处理后的图像
    ax.plot(x, y_act, label=f"{title}(x)", color="#1f77b4", linewidth=2)

    # 设置标题、坐标轴范围与网格
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)  # X轴参考线
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)  # Y轴参考线
    ax.legend(loc="upper left")

plt.tight_layout()
plt.show()
