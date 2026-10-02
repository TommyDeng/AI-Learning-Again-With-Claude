from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=200, n_features=10, random_state=0)

model = RandomForestClassifier(n_estimators=100, random_state=0)

# 5折交叉验证:把数据分成5份,轮流用4份训练、1份验证,共做5次
# 交叉验证不会改变模型本身
scores = cross_val_score(model, X, y, cv=5)
print("每折准确率:", scores)
print("平均准确率:", scores.mean(), "±", scores.std())


# 为什么交叉验证更可靠: 单次 train_test_split 的结果可能因为"运气好/坏"而偏高或偏低，交叉验证跑5次取平均,结果更稳定,也能看出模型表现的波动程度(标准差)。
