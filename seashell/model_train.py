import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from preprocession import data_scaled
# 生成不同k值的WCSS
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)


# 绘制WCSS曲线
plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, 11), wcss, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.show()