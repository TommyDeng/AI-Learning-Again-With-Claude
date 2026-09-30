import pandas as pd

# 创建DataFrame
data = {
    "姓名": ["小明", "小红", "小李"],
    "年龄": [25, 22, 30],
    "城市": ["台北", "高雄", "台中"],
}
df = pd.DataFrame(data)
print(df)

# 查看数据
print(df.head())  # 前5行
print(df.info())  # 数据类型概览
print(df.describe())  # 统计摘要

# 筛选
print(df[df["年龄"] > 24])  # 条件筛选
print(df["姓名"])  # 取一列
print(df.loc[0])  # 取一行(按标签)
print(df.iloc[0])  # 取一行(按位置)

# 新增/修改列
df["年龄+1"] = df["年龄"] + 1

# 分组统计
print(df.groupby("城市")["年龄"].mean())

# 读写文件
# df.to_csv("data.csv", index=False)
# df = pd.read_csv("data.csv")

# 处理缺失值
df2 = pd.DataFrame({"x": [1, None, 3]})
print(df2.isnull())  # 检查缺失值
df2 = df2.dropna()  # 删除缺失行
# df2 = df2.fillna(0)        # 或用0填充
