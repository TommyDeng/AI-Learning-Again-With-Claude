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

#  Pandas/NumPy数据结构对比
#
#   结构                更贴切的数据库 / Excel 比喻                      数学 / 现实世界比喻                    特征
# Pandas DataFrame      整个 Excel 工作表 / SQL 数据表                  结构化业务账本                       有列名、有行索引、类型可不同
# Pandas Series         Excel 中的“单独一列”                            带标签的一维数据序列                 有列名、有行索引、类型统一
# NumPy ndarray         去除所有行列标签后的“纯数字网格”                  标量 / 向量 / 矩阵 / 张量           无列名、纯数字位置坐标、全同质类型
