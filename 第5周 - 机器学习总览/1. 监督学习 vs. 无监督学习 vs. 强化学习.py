# 监督学习:有标签数据,学习输入到输出的映射
# 例如:给定房屋面积(X),预测房价(y) —— 你已经在第3、4周做过了

# 无监督学习:没有标签,让模型自己发现数据结构
# 例如:聚类,把相似的数据点分到一组

# 强化学习:智能体通过试错,根据奖励信号学习最优策略
# 例如:游戏AI(后面第20周左右会接触)

# 用代码直观感受"无监督学习"是什么样子:
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Set CJK-compatible font (Microsoft YaHei for Windows)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]

# Fix minus sign display issues
plt.rcParams["axes.unicode_minus"] = False


# 生成两簇数据,但不告诉模型标签
np.random.seed(0)
cluster1 = np.random.randn(50, 2) + [2, 2]
cluster2 = np.random.randn(50, 2) + [8, 8]
X = np.vstack([cluster1, cluster2])

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c="red",
    marker="x",
    s=200,
)
plt.title("KMeans聚类结果")
plt.show()

# 模型完全没被告知哪个点属于哪一类，纯靠数据本身的分布把它们分开了——这就是无监督学习。
