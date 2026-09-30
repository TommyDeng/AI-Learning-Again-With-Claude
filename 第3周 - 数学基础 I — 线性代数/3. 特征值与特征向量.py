import numpy as np

A = np.array([[4, 2], [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("特征值:", eigenvalues)
print("特征向量:\n", eigenvectors)

# 直觉理解（不用深究证明）: 特征向量是矩阵变换时"方向不变，只被拉伸"的那些特殊向量，特征值就是拉伸的倍数。
# 这个概念在PCA降维、协方差矩阵分析中会用到，现在有个印象即可。
