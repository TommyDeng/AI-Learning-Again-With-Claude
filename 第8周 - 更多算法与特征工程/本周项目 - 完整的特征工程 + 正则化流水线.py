import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# 构造一个新特征:房间数/卧室数比例
X["房间卧室比"] = X["AveRooms"] / X["AveBedrms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化(先fit训练集,再用同样的参数transform测试集 —— 切记不能用测试集数据去fit!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# 查看特征重要性(系数绝对值越大,影响越大 —— 前提是已经标准化过)
coef_df = pd.DataFrame({"特征": X.columns, "系数": model.coef_}).sort_values(
    "系数", key=abs, ascending=False
)
print("\n特征重要性排序:\n", coef_df)


# 这里有个关键细节值得强调: scaler.fit_transform(X_train) 和 scaler.transform(X_test)——训练集用 fit_transform，测试集只用 transform。
# 如果测试集也用 fit，相当于让模型"偷看"了测试集的统计信息(均值、标准差)，评估结果会虚高，这是数据泄露(data leakage)的一种常见形式，实际项目中是大忌。
