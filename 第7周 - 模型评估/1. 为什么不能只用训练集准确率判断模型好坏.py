from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=200, n_features=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 故意让决策树深度不受限制 —— 容易过拟合
overfit_tree = DecisionTreeClassifier(max_depth=None, random_state=0)
overfit_tree.fit(X_train, y_train)

print("训练集准确率:", overfit_tree.score(X_train, y_train))  # 很可能接近1.0
print("测试集准确率:", overfit_tree.score(X_test, y_test))  # 会明显更低


# 这就是过拟合：模型把训练集的细节甚至噪声都"背"下来了，看起来训练集表现完美，但遇到没见过的数据就露馅。这也是为什么必须留出测试集。
