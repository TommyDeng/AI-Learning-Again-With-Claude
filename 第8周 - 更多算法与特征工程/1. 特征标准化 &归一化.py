import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 模拟一个特征量纲差异很大的场景:年龄(0-100) vs 收入(0-1000000)
X = np.array([[25, 50000], [35, 80000], [45, 120000], [55, 60000]])

# 标准化(Z-score):转换成均值0、标准差1
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
print("标准化后:\n", X_standardized)

# 归一化(Min-Max):压缩到[0,1]区间
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)
print("归一化后:\n", X_normalized)
