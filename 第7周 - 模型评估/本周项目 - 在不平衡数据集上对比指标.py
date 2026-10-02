from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

# 制造一个明显不平衡的数据集(95% vs 5%)
X, y = make_classification(
    n_samples=1000, n_features=10, weights=[0.95, 0.05], random_state=0
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
# stratify=y 保证训练/测试集里正负样本比例一致,这点很重要

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("准确率:", model.score(X_test, y_test))
print("\n完整分类报告:\n", classification_report(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))

# 用交叉验证,并指定用F1而不是准确率作为评估标准
scores = cross_val_score(model, X, y, cv=5, scoring="f1")
print("\n5折交叉验证F1分数:", scores)
print("平均F1:", scores.mean())


# 跑完这个项目，你会直观看到：classification_report 里准确率看起来可能还不错，但少数类(5%那一类)的 precision/recall
# 可能很差——这正是只看准确率会踩的坑，也是为什么工业界评估模型时几乎从不只看一个指标。
