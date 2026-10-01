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


# Python 官方在语法层面上压根就没有提供原生的“数组（Array）”类型，你在 Python 里用中括号 [...] 写出来的，全部都是列表（list）！
# 真实的 Python 数组有两种方法创建：
# 1. NumPy 数组（数据科学与 AI 标准）
# 2. Python 内置的 array 模块（较少使用）


# 数组的概念：
# 中文里将 Array 翻译成 “数组”，常常会给刚接触编程的人带来字面上的误解，让人以为它里面装的一定是“数字/数值”。
# 但实际上：“数”字是翻译的历史遗留：Array 在英语中的本意是 “阵列”、“排列”、“一排”（比如“一排士兵”叫 An array of soldiers）。
# 在引进计算机术语时被翻译成了“数组”，这里的“数”字更偏向于“数目/一排数据”的意思，而不是单纯指“数字/数值”。
# 它的核心是“数据（Data）”：无论里面装的是字符（'a'）、文本（"hello"）、逻辑值（True）、图片像素，还是你自定义的复杂对象，只要它们被按顺序、有编号地整齐排列在内存中，这个结构就叫 Array（数组/阵列）。
# 所以，在计算机科学中：
# “数组”里的“数” ≠ 数值（Number）
# “数组”里的“数” = 数据（Data / Item)


# 1维数组 → 向量（Vector）
# 2维数组 → 矩阵（Matrix）
# 3维及更高维数组 → 高维张量（High-order Tensor）（或者直接统称“张量”）
