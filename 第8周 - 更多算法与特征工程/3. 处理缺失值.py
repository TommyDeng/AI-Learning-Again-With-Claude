import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame(
    {"年龄": [25, np.nan, 35, 40, np.nan], "收入": [50000, 60000, np.nan, 80000, 55000]}
)

print("缺失值统计:\n", df.isnull().sum())

# 用均值填充
imputer = SimpleImputer(strategy="mean")
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print(df_filled)

# 其他策略: strategy='median'(中位数,对异常值更稳健) 或 'most_frequent'(众数,适合类别特征)
