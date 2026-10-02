from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, random_state=0)

# 先分出测试集(最终才用一次,绝不参与任何调参)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 再把剩下的分成训练集和验证集(调参时反复使用)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=0
)

print("训练集:", len(X_train), " 验证集:", len(X_val), " 测试集:", len(X_test))


# 三者的分工:

# 训练集 — 模型学习参数用的
# 验证集 — 调参数、选模型用的（比如试哪个 max_depth 效果最好）
# 测试集 — 只在最后用一次，模拟"模型真正上线后遇到的新数据"
