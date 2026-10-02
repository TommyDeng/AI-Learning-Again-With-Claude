import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# 设置绘图风格与中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 生成带有噪声与无效特征的数据集
# 总共 10 个特征，只有 3 个特征有真正贡献 (n_informative=3)
X, y, true_coef = make_regression(
    n_samples=100, n_features=10, n_informative=3, noise=10, coef=True, random_state=42
)

# 2. 数据标准化（正则化模型必须做标准化，保证各特征惩罚尺度一致）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 实例化模型并训练
# alpha 控制正则化强度 (\lambda)
alpha_value = 1.0

# L1 正则化 (Lasso)
lasso = Lasso(alpha=alpha_value, random_state=42)
lasso.fit(X_scaled, y)

# L2 正则化 (Ridge)
ridge = Ridge(alpha=alpha_value, random_state=42)
ridge.fit(X_scaled, y)

# 4. 打印对比真正的系数与两个模型学到的权重系数 w
df_coef = pd.DataFrame(
    {
        "真实权重 (True)": np.round(true_coef, 2),
        "L1 (Lasso) 权重": np.round(lasso.coef_, 2),
        "L2 (Ridge) 权重": np.round(ridge.coef_, 2),
    }
)

print("=== 权重系数 w 对比表 ===")
print(df_coef.to_string())

# 5. 使用 Matplotlib 绘制权重系数对比柱状图
plt.figure(figsize=(10, 6))
indices = np.arange(len(true_coef))
width = 0.25

plt.bar(
    indices - width,
    true_coef,
    width=width,
    label="真实权重 (True)",
    color="gray",
    alpha=0.6,
)
plt.bar(indices, lasso.coef_, width=width, label="L1 (Lasso) 权重", color="crimson")
plt.bar(
    indices + width,
    ridge.coef_,
    width=width,
    label="L2 (Ridge) 权重",
    color="royalblue",
)

plt.title(
    f"L1 (Lasso) 与 L2 (Ridge) 权重系数对比 (alpha={alpha_value})",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("特征索引 (Feature Index)", fontsize=11)
plt.ylabel("权重系数 $w$ 强度", fontsize=11)
plt.xticks(indices, [f"X_{i}" for i in range(len(true_coef))])
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# 直觉理解: 正则化是在损失函数里加一个"惩罚项"，惩罚权重过大。这样模型就不会为了完美拟合训练集而把某个权重调得特别极端——权重越极端，模型越容易对训练数据里的噪声过度敏感。

# L2(Ridge) — 让所有权重都变小一点，比较"温和"
# L1(Lasso) — 会把不重要特征的权重直接压到0，相当于自动做了特征选择
