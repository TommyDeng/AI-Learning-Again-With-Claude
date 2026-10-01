from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

X, y = make_classification(n_samples=200, n_features=4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

svm = SVC(kernel="rbf")
svm.fit(X_train, y_train)
print("SVM准确率:", svm.score(X_test, y_test))

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("kNN准确率:", knn.score(X_test, y_test))

# 直觉理解:

# SVM 试图找到一个"间隔最大"的分界线/分界面来区分类别
# k-NN 更简单粗暴:看一个新数据点周围最近的k个邻居里，多数是什么类别，就判它是什么类别——完全不需要"训练"，预测时才计算
