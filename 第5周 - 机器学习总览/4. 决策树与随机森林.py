from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=200, n_features=4, random_state=0)

# 单一决策树
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X, y)
print("决策树准确率:", tree.score(X, y))

# 随机森林(多个决策树投票)
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X, y)
print("随机森林准确率:", forest.score(X, y))

# 直觉理解:
# 决策树就是不断问"这个特征是否大于某个值"来划分数据，像玩20问游戏。
# 随机森林是训练很多棵"看到不同数据子集"的树，然后投票决定最终结果——多个不完美的模型组合起来，往往比单个模型更稳定、更准。
