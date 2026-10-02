import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans

# 加载 Iris 数据集与评估工具
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 引入 7 种算法
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 设置绘图风格与字体（防止中文乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 加载 Iris 原始数据
iris = load_iris()
X_raw = iris.data
y_raw = iris.target

# ==========================================
# 第一部分：6 种分类与聚类算法（使用前 2 个特征画决策边界）
# ==========================================

# 1. 提取前两个特征：花萼长度 (Sepal Length) 与 花萼宽度 (Sepal Width)
X_cls = X_raw[:, :2]
y_cls = y_raw

# 2. 数据标准化（基于距离/梯度的算法必需）
scaler = StandardScaler()
X_cls_scaled = scaler.fit_transform(X_cls)

# 3. 实例化 6 种算法
cls_models = {
    "逻辑回归": LogisticRegression(max_iter=200),
    "决策树": DecisionTreeClassifier(max_depth=3, random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF核)": SVC(kernel="rbf", probability=True),
    "K-Means (无监督, K=3)": KMeans(n_clusters=3, random_state=42, n_init=10),
}

# 创建网格范围以绘制决策边界
x_min, x_max = X_cls_scaled[:, 0].min() - 0.5, X_cls_scaled[:, 0].max() + 0.5
y_min, y_max = X_cls_scaled[:, 1].min() - 0.5, X_cls_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# 4. 绘图：画出 6 种算法在 Iris 上的分类边界与效果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (name, model) in enumerate(cls_models.items()):
    ax = axes[i]

    if name == "K-Means (无监督, K=3)":
        # 无监督训练：不传入 y 标签
        model.fit(X_cls_scaled)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 将簇编号（0, 1, 2）映射回真实标签计算近似准确率
        cluster_labels = model.labels_
        mapped_preds = np.zeros_like(cluster_labels)
        for cluster in range(3):
            mask = cluster_labels == cluster
            if np.sum(mask) > 0:
                mapped_preds[mask] = mode(y_cls[mask], keepdims=False).mode
        acc = accuracy_score(y_cls, mapped_preds)
        score_text = f"簇匹配度: {acc * 100:.1f}%"
    else:
        # 监督训练：传入特征与真实标签
        model.fit(X_cls_scaled, y_cls)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        acc = accuracy_score(y_cls, model.predict(X_cls_scaled))
        score_text = f"准确率: {acc * 100:.1f}%"

    # 绘制背景决策边界区域
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    # 绘制 Iris 真实样本散点
    scatter = ax.scatter(
        X_cls_scaled[:, 0],
        X_cls_scaled[:, 1],
        c=y_cls,
        cmap=plt.cm.Set1,
        edgecolors="k",
        s=40,
    )

    ax.set_title(f"{name}\n({score_text})", fontsize=12)
    ax.set_xlabel("花萼长度 (标准化)")
    ax.set_ylabel("花萼宽度 (标准化)")

plt.suptitle(
    "Iris 数据集：6 种分类/聚类算法决策边界可视化", fontsize=16, fontweight="bold"
)
plt.tight_layout()
plt.show()

# ==========================================
# 第二部分：线性回归算法（预测花瓣宽度）
# ==========================================

# 取第 3 个特征（花瓣长度 Petal Length）预测第 4 个特征（花瓣宽度 Petal Width）
X_reg = X_raw[:, 2:3]  # 特征
y_reg = X_raw[:, 3]  # 目标

# 训练线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X_reg, y_reg)
y_pred_reg = lin_reg.predict(X_reg)
r2 = r2_score(y_reg, y_pred_reg)

# 绘制线性回归拟合图像
plt.figure(figsize=(8, 5))
plt.scatter(
    X_reg, y_reg, color="darkorange", alpha=0.8, edgecolors="k", label="Iris 真实数据点"
)
plt.plot(
    X_reg,
    y_pred_reg,
    color="navy",
    linewidth=2.5,
    label=f"线性回归拟合线 ($R^2 = {r2:.2f}$)",
)
plt.title(
    "Iris 数据集：线性回归 (用花瓣长度预测花瓣宽度)", fontsize=14, fontweight="bold"
)
plt.xlabel("花瓣长度 (Petal Length / cm)")
plt.ylabel("花瓣宽度 (Petal Width / cm)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
