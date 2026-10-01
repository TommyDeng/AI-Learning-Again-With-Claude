from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 加载经典数据集:根据花瓣/花萼长宽,分类鸢尾花品种
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("测试集准确率:", accuracy_score(y_test, y_pred))

# 看看哪个特征最重要
for name, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"{name}: {importance:.4f}")


# 这是机器学习界的"Hello World"，几乎每个人入门都会做这个数据集。
# 跑通后你就完成了第一个完整的分类项目：加载数据 → 划分训练/测试集 → 训练模型 → 预测 → 评估。
