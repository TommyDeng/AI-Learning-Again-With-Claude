import numpy as np
import pandas as pd

# 模拟一个学生成绩数据集
data = {
    "姓名": ["小明", "小红", "小李", "小陈", "小周"],
    "数学": [85, 92, 78, 90, 65],
    "英语": [70, 88, 95, 60, 75],
}
df = pd.DataFrame(data)

# 基本分析
df["总分"] = df["数学"] + df["英语"]
df["平均分"] = df["总分"] / 2

print("成绩最高的学生:")
print(df.loc[df["总分"].idxmax()])

print("\n数学成绩标准差:", np.std(df["数学"]))
print("按总分排序:")
print(df.sort_values("总分", ascending=False))
