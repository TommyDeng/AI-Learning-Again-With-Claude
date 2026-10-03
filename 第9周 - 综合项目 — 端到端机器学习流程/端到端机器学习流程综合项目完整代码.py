import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

##Step 1:加载数据 + 探索性分析(EDA)
print("Step 1:加载数据 + 探索性分析(EDA)")
# 泰坦尼克数据集,seaborn自带,不用额外下载
df = sns.load_dataset("titanic")

print(df.shape)
print(df.head())

# info() 看的是“数据结构与元数据”（宏观结构、数据类型、缺失值情况、内存占用）。
print(df.info())

# describe() 看的是“数据的数值统计特征”（微观分布、均值、分位数、极值等）。
print(df.describe())

# 查看目标变量分布
print("\n生存情况分布:\n", df["survived"].value_counts())

# 查看缺失值
print("\n缺失值统计:\n", df.isnull().sum().sort_values(ascending=False))


# 简单可视化,看看哪些特征可能和生存率相关
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.barplot(data=df, x="sex", y="survived", ax=axes[0])
axes[0].set_title("性别 vs 生存率")

sns.barplot(data=df, x="pclass", y="survived", ax=axes[1])
axes[1].set_title("舱位等级 vs 生存率")

sns.histplot(data=df, x="age", hue="survived", multiple="fill", ax=axes[2])
axes[2].set_title("年龄分布 vs 生存情况")

plt.tight_layout()
plt.show()

##EDA的意义: 建模前先用眼睛看数据，往往能发现模型训练时不容易一眼看出的规律
# ——比如这里你会看到女性、头等舱乘客生存率明显更高，这些观察会指导你后面怎么做特征工程。

## Step 2:数据清洗
print("\nStep 2:数据清洗")
# 丢弃缺失太多、或对预测没帮助的列
df_clean = df.drop(
    columns=["deck", "embark_town", "alive", "who", "adult_male", "class"]
)

# 年龄缺失:用中位数填充(比均值更抗异常值)
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())

# 登船港口缺失:用众数填充(只有2个缺失值)
df_clean["embarked"] = df_clean["embarked"].fillna(df_clean["embarked"].mode()[0])

print("清洗后缺失值:\n", df_clean.isnull().sum())

## Step 3:特征工程
print("\nStep 3:特征工程")
# 类别特征编码
df_clean["sex"] = df_clean["sex"].map({"male": 0, "female": 1})
df_clean = pd.get_dummies(df_clean, columns=["embarked"], drop_first=True)

# 构造新特征:家庭成员总数(配偶/子女 + 兄弟姐妹/父母)
df_clean["family_size"] = df_clean["sibsp"] + df_clean["parch"] + 1

# 构造新特征:是否独自一人
df_clean["is_alone"] = (df_clean["family_size"] == 1).astype(int)

print(df_clean.head())
print(df_clean.dtypes)

# 处理剩下不是数值的列(比如alone这种布尔列)
df_clean["alone"] = df_clean["alone"].astype(int)

# 确认全部是数值类型,可以进模型了
print("\n清洗后数据类型:\n")
print(df_clean.dtypes)


# Step 4:划分数据 + 训练多个模型对比
print("\nStep 4:划分数据 + 训练多个模型对比 - 横向对比不同算法的表现")

X = df_clean.drop(columns=["survived"])
y = df_clean["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
# fit_transform()相当于：先计算均值和标准差fit(),再进行标准化transform()
# 如果分开写 fit(X) 然后 transform(X) 更便于理解，但是会进行两次遍历，损失性能。
X_train_scaled = scaler.fit_transform(X_train)
# transform()只进行标准化，不会重新计算均值和标准差。
X_test_scaled = scaler.transform(X_test)

models = {
    "逻辑回归": LogisticRegression(max_iter=1000),
    "随机森林": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf"),
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"\n=== {name} ===")
    print(f"准确率: {acc:.4f}")
    print(classification_report(y_test, pred))

# Step 5:用交叉验证+网格搜索调参
print("\nStep 5:用交叉验证+网格搜索调参 - 纵向挖掘单一算法的潜力")

# 以随机森林为例,搜索最佳超参数组合
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,  # 用所有CPU核心并行搜索,加快速度
)

grid_search.fit(X_train_scaled, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳交叉验证准确率:", grid_search.best_score_)

# 用找到的最佳模型在测试集上最终评估
best_model = grid_search.best_estimator_
test_acc = best_model.score(X_test_scaled, y_test)
print("测试集准确率:", test_acc)

# GridSearchCV 在做什么:
# 它会把 param_grid 里所有参数组合都试一遍（3×4×3=36种组合），每种组合都做5折交叉验证，自动挑出平均表现最好的那组参数。
# 这比你手动一个个试参数高效得多，是实际项目调参的标准做法。


# Step 6:解读结果 — 哪些特征真正重要
print("\nStep 6:解读结果 — 哪些特征真正重要")

importance_df = pd.DataFrame(
    {"特征": X.columns, "重要性": best_model.feature_importances_}
).sort_values("重要性", ascending=False)

print(importance_df)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x="重要性", y="特征")
plt.title("特征重要性排序")
plt.show()

# 你大概率会看到 sex(性别)、fare(票价,间接反映舱位)、age 排在前列——这和Step 1的EDA图表观察到的规律互相印证，
# 这种"EDA发现规律 → 模型验证规律"的闭环，正是一个靠谱数据科学项目该有的样子。
