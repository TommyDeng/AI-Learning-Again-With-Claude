import numpy as np

# 向量就是一维数组
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 向量加减
print(v1 + v2)  # [5 7 9]
print(v1 - v2)  # [-3 -3 -3]

# 数乘
print(v1 * 2)  # [2 4 6]

# 点积(内积)— 深度学习中极其常用
dot = np.dot(v1, v2)
print(dot)  # 1*4 + 2*5 + 3*6 = 32

# 向量的模(长度)
norm = np.linalg.norm(v1)
print(norm)  # sqrt(1^2+2^2+3^2)

# 单位向量
unit_v1 = v1 / norm
print(unit_v1)

# 直觉理解: 点积衡量两个向量"方向的相似程度"。这个概念后面在注意力机制(attention)、词向量相似度计算里会反复出现。
