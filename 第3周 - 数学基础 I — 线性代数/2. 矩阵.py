import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 矩阵加法
print(A + B)

# 矩阵乘法(注意:不是对应元素相乘!)
print(A @ B)  # 或 np.matmul(A, B)
print(A * B)  # 这是逐元素相乘,和矩阵乘法不同,注意区分

# 转置
print(A.T)

# 单位矩阵
I = np.eye(2)
print(I)

# 逆矩阵
A_inv = np.linalg.inv(A)
print(A_inv)
print(A @ A_inv)  # 应该约等于单位矩阵

# 行列式
print(np.linalg.det(A))

# 为什么矩阵乘法这么重要: 神经网络的每一层，本质上都是"输入向量 × 权重矩阵 + 偏置"，然后过一个激活函数。
# 理解 A @ B 就是理解神经网络前向传播的核心运算
