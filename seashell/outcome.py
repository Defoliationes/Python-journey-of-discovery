import matplotlib.pyplot as plt
from preprocession import result, data_scaled, df
from sklearn.cluster import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取聚类结果
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
labels = kmeans.fit_predict(data_scaled)
result['Cluster'] = labels
df['Cluster'] = labels

file_path = 'result_df.csv'
df.to_csv(file_path, index=False, encoding='utf_8_sig')

# 重新绘制3D散点图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']
# 重新为每个聚类绘制数据点
for i, cluster in enumerate(sorted(result['Cluster'].unique())):
    cluster_data = result[result['Cluster'] == cluster]
    ax.scatter(cluster_data['楼层'], cluster_data['住房面积'], cluster_data['价格（元/月）'],
               c=colors[i], label=f'Cluster {cluster}', alpha=0.6)
ax.set_xlabel('楼层')
ax.set_ylabel('住房面积')
ax.set_zlabel('价格（元/月）')
ax.legend() # 显示图例
plt.show() # 显示图形