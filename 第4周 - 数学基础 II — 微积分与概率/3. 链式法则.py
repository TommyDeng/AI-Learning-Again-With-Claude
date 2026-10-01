# （反向传播的数学基础）
# 为什么重要: 反向传播本质上就是层层应用链式法则，把"最终损失对每一层参数的影响"一路传回去。

# 假设 y = f(g(x)), 链式法则: dy/dx = dy/dg * dg/dx


def g(x):
    return x**2  # g(x) = x^2


def f(g_val):
    return g_val + 3  # f(g) = g + 3


x = 2

# 数值验证
h = 1e-5
y1 = f(g(x + h))
y2 = f(g(x - h))
dy_dx = (y1 - y2) / (2 * h)
print("dy/dx =", dy_dx)  # 应该约等于 4 (因为 y=(x^2+3), dy/dx=2x=4)
