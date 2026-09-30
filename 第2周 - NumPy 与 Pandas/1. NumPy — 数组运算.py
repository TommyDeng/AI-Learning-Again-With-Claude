# pip install numpy

import numpy as np

# 创建数组
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2], [3, 4]])  # 二维数组(矩阵)

print(a.shape)  # (5,)
print(b.shape)  # (2, 2)

# 向量化运算(比Python原生循环快很多)
print(a + 10)  # [11 12 13 14 15]
print(a * 2)  # [2 4 6 8 10]

# 矩阵运算
c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])
print(c + d)  # 矩阵加法
print(c @ d)  # 矩阵乘法(重要!深度学习核心运算)
print(c.T)  # 转置

# 常用函数
print(np.mean(a))  # 均值
print(np.std(a))  # 标准差
print(np.zeros((2, 3)))  # 全0矩阵
print(np.ones((2, 3)))  # 全1矩阵
print(np.random.rand(2, 3))  # 随机矩阵

# 索引与切片(和列表类似,但更强大)
print(b[0, 1])  # 第0行第1列
print(b[:, 0])  # 所有行的第0列
