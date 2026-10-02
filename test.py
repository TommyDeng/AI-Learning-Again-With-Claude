import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans

# Scikit-Learn 数据生成与预处理
from sklearn.datasets import make_blobs, make_regression
from sklearn.ensemble import RandomForestClassifier

# 引入 7 种算法
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 设置绘图风格与中文显示（防止中文乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 第一部分：6 种分类与聚类算法可视化演示
# ==========================================

# 1. 生成二维分类数据集（方便画图展示决策边界）
X_cls, y_cls = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=1.5)

# 2. 数据标准化（基于距离/梯度的算法必需）
scaler = StandardScaler()
X_cls_scaled = scaler.fit_transform(X_cls)

# 3. 实例化 6 种算法（分类与聚类）
cls_models = {
    "逻辑回归": LogisticRegression(),
    "决策树": DecisionTreeClassifier(max_depth=3, random_state=42),
    "随机森林": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF核)": SVC(kernel="rbf", probability=True),
    "K-Means (无监督)": KMeans(n_clusters=2, random_state=42, n_init=10),
}

# 创建网格用于绘制决策边界
x_min, x_max = X_cls_scaled[:, 0].min() - 0.5, X_cls_scaled[:, 0].max() + 0.5
y_min, y_max = X_cls_scaled[:, 1].min() - 0.5, X_cls_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# 4. 绘图：画出 6 种算法的决策边界与拟合效果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (name, model) in enumerate(cls_models.items()):
    ax = axes[i]

    if name == "K-Means (无监督)":
        # 无监督训练：不传入 y 标签
        model.fit(X_cls_scaled)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 将簇编号映射回对应的分类标签计算近似准确率
        cluster_labels = model.labels_
        mapped_preds = np.zeros_like(cluster_labels)
        for cluster in range(2):
            mask = cluster_labels == cluster
            if np.sum(mask) > 0:
                mapped_preds[mask] = mode(y_cls[mask], keepdims=False).mode
        acc = accuracy_score(y_cls, mapped_preds)
        score_text = f"匹配度: {acc * 100:.1f}%"
    else:
        # 监督训练：传入 X 和 y
        model.fit(X_cls_scaled, y_cls)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        acc = accuracy_score(y_cls, model.predict(X_cls_scaled))
        score_text = f"准确率: {acc * 100:.1f}%"

    # 绘制背景决策边界区域
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    # 绘制样本散点
    scatter = ax.scatter(
        X_cls_scaled[:, 0],
        X_cls_scaled[:, 1],
        c=y_cls,
        cmap=plt.cm.coolwarm,
        edgecolors="k",
        s=40,
    )

    ax.set_title(f"{name}\n({score_text})", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("6 种分类/聚类算法的决策边界与结果演示", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

# ==========================================
# 第二部分：线性回归算法可视化演示
# ==========================================

# 1. 生成一维回归数据集
X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# 2. 训练线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(X_reg, y_reg)
y_pred_reg = lin_reg.predict(X_reg)
r2 = r2_score(y_reg, y_pred_reg)

# 3. 绘制拟合直线图
plt.figure(figsize=(8, 5))
plt.scatter(X_reg, y_reg, color="blue", alpha=0.6, label="真实数据点")
plt.plot(
    X_reg, y_pred_reg, color="red", linewidth=2, label=f"拟合直线 ($R^2 = {r2:.2f}$)"
)
plt.title("线性回归 (Linear Regression) 拟合演示", fontsize=14, fontweight="bold")
plt.xlabel("特征 X")
plt.ylabel("目标 y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
