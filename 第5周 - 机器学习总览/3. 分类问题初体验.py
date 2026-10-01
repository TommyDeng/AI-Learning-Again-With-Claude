from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# 生成一个二分类数据集
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=0
)

model = LogisticRegression()
model.fit(X, y)

print("准确率:", model.score(X, y))
print("前5个预测:", model.predict(X[:5]))
print("前5个真实标签:", y[:5])

# 回归 vs. 分类的关键区别:
# 回归预测一个连续数值(比如房价、温度)，分类预测一个离散类别(比如"是/否"、"猫/狗/鸟")。
# 逻辑回归这个名字带"回归"但其实是分类算法——这是个常见的历史命名坑，注意别搞混。
