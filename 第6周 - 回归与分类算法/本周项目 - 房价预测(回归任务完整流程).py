from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 加载加州房价数据集(经典回归入门数据集)
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 对比两个模型
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print(
    "线性回归: MSE =",
    mean_squared_error(y_test, lr_pred),
    " R² =",
    r2_score(y_test, lr_pred),
)
print(
    "随机森林: MSE =",
    mean_squared_error(y_test, rf_pred),
    " R² =",
    r2_score(y_test, rf_pred),
)

# R² 分数(决定系数) 是回归任务里一个很常用的指标,越接近1说明模型解释数据变化的能力越强。
# 你会发现随机森林通常比简单线性回归表现更好,因为房价和特征之间不是简单的线性关系。
