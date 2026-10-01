import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# 划分训练集和测试集(下周会细讲为什么要划分)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

print("斜率 (w):", model.coef_)
print("截距 (b):", model.intercept_)

# 预测
y_pred = model.predict(X_test)
print("前5个预测值:", y_pred[:5].flatten())
print("前5个真实值:", y_test[:5].flatten())


# 你会发现：model.fit() 这一行代码，内部做的事情本质上就是你前两周手写的"梯度下降"或"矩阵解析解"——只是被封装好了，速度更快、更稳健。
# 理解底层原理后再用库，你才知道调参调的是什么。
