import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# 模拟一个不平衡的分类场景:比如100个样本里只有5个是"患病"(正类)
y_true = [0] * 95 + [1] * 5
y_pred = [0] * 97 + [1] * 3  # 模型预测:97个阴性,3个阳性(但预测不一定对位)

# 构造一个更真实的例子
np.random.seed(0)
y_true = np.array([0] * 90 + [1] * 10)
y_pred = np.array([0] * 95 + [1] * 5)  # 模型几乎总预测"阴性"

print("准确率:", accuracy_score(y_true, y_pred))  # 看起来很高,但具有迷惑性!
print("精确率:", precision_score(y_true, y_pred, zero_division=0))
print("召回率:", recall_score(y_true, y_pred, zero_division=0))
print("F1分数:", f1_score(y_true, y_pred, zero_division=0))
print("混淆矩阵:\n", confusion_matrix(y_true, y_pred))
