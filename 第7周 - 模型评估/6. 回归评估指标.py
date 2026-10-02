import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.5])
y_pred = np.array([2.8, 5.2, 2.0, 6.5, 5.0])

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("MSE:", mse)  # 均方误差,对大误差惩罚更重(因为平方了)
print("RMSE:", rmse)  # 开根号后单位和y一致,更好解读
print("MAE:", mae)  # 平均绝对误差,对异常值没那么敏感
print("R²:", r2)  # 模型解释了多少比例的数据变化,1最好
