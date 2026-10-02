import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split

# 制造一个有很多无关特征的数据集
X, y = make_regression(
    n_samples=100, n_features=20, n_informative=5, noise=10, random_state=0
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 普通线性回归(无正则化)
lr = LinearRegression()
lr.fit(X_train, y_train)
print("普通线性回归 R²:", lr.score(X_test, y_test))

# L2正则化(Ridge):让权重整体变小,但不会变成0
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print("Ridge (L2) R²:", ridge.score(X_test, y_test))

# L1正则化(Lasso):会把不重要特征的权重直接压成0,起到特征选择的作用
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
print("Lasso (L1) R²:", lasso.score(X_test, y_test))

# 看看Lasso把哪些特征的权重压成了0
print("\nLasso系数(0表示该特征被忽略):")
print(lasso.coef_)
print("非零特征数量:", np.sum(lasso.coef_ != 0), "/ 总特征数:", len(lasso.coef_))
