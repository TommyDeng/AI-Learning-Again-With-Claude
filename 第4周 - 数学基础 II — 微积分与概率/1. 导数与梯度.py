# 手动实现数值微分,理解导数的本质:变化率
def f(x):
    return x**2


def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


print(numerical_derivative(f, 3))  # 应该接近 6 (因为 d/dx(x^2) = 2x, 在x=3时=6)
