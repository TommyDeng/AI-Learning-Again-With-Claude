from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

print("聚类中心:\n", kmeans.cluster_centers_)
print("每个样本被分配的簇标签(前10个):", kmeans.labels_[:10])
