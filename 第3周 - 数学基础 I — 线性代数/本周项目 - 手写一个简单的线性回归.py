import numpy as np

# 生成模拟数据: y = 2x + 1 + 噪声
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# 加一列1,用于计算截距(偏置项)
X_b = np.hstack([np.ones((100, 1)), X])  # shape: (100, 2)

# 正规方程求解: theta = (X^T X)^-1 X^T y
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print("截距:", theta[0][0])  # 应该接近1
print("斜率:", theta[1][0])  # 应该接近2


# 这个项目能让你真正"看见"线性代数怎么用在机器学习里

# 这行代码 theta = (X^T X)^-1 X^T y 就是线性回归的解析解公式，完全靠今天学的转置、矩阵乘法、逆矩阵算出来的——没有用任何机器学习库。
