import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=300, n_features=10, weights=[0.9, 0.1], random_state=0
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]  # 取"预测为正类"的概率

auc = roc_auc_score(y_test, y_prob)
print("AUC:", auc)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # 随机猜测基准线
plt.xlabel("假阳性率 (FPR)")
plt.ylabel("真阳性率 (TPR)")
plt.title("ROC曲线")
plt.legend()
plt.show()

# 直觉理解: AUC 衡量"模型把正类排得比负类靠前"的整体能力，取值0.5(等同瞎猜)到1.0(完美分类)。它不依赖你选择哪个分类阈值，比单纯看准确率更全面。
