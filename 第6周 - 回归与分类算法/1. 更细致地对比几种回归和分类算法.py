from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ---回归任务对比---
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models_reg = {
    "线性回归": LinearRegression(),
    "决策树回归": DecisionTreeRegressor(max_depth=5, random_state=0),
    "随机森林回归": RandomForestRegressor(n_estimators=100, random_state=0),
}

print("=== 回归任务对比 ===")
for name, model in models_reg.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    print(f"{name}: MSE = {mse:.2f}")

# ---分类任务对比---
X, y = make_classification(n_samples=200, n_features=5, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

models_clf = {
    "逻辑回归": LogisticRegression(),
    "决策树分类": DecisionTreeClassifier(max_depth=5, random_state=0),
    "随机森林分类": RandomForestClassifier(n_estimators=100, random_state=0),
}

print("\n=== 分类任务对比 ===")
for name, model in models_clf.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"{name}: 准确率 = {acc:.4f}")
