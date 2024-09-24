import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# 生成示例数据
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=30)
data = {date: np.random.poisson(lam=20, size=24) for date in dates}
df = pd.DataFrame(data).T

# 新的一天数据
new_day = pd.Series(np.random.poisson(lam=20, size=24), name='2023-01-31')

# 添加新一天数据到DataFrame
df = pd.concat([df, new_day.to_frame().T])

# 数据标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 使用K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)  # 调整聚类数
kmeans.fit(df_scaled[:-1])
new_day_cluster = kmeans.predict(df_scaled[-1].reshape(1, -1))

# 使用DBSCAN聚类，调整eps参数确保至少10%的异常比例
dbscan = DBSCAN(eps=0.5, min_samples=2)  # 初始参数
while True:
    dbscan.fit(df_scaled)
    new_day_dbscan = dbscan.fit_predict(df_scaled[-1].reshape(1, -1))
    labels = dbscan.labels_
    # 检查异常点比例
    num_anomalies = np.sum(labels == -1)
    anomaly_ratio = num_anomalies / len(labels)
    if anomaly_ratio >= 0.1:
        break
    dbscan.eps += 0.1  # 增加eps以增加异常点比例

# 可视化K-means聚类结果
plt.figure(figsize=(15, 5))
plt.plot(new_day, label='New Day Traffic', marker='o')
plt.title('Traffic Analysis for New Day (K-means Clustering)')
plt.xlabel('Hour')
plt.ylabel('Traffic')
plt.grid(True)
anomalies_kmeans = []
for i, val in enumerate(new_day):
    if kmeans.predict(df_scaled[i].reshape(1, -1)) != new_day_cluster:
        anomalies_kmeans.append(i)
        plt.scatter(i, val, color='red')
plt.legend()
plt.show()

# 可视化DBSCAN聚类结果
plt.figure(figsize=(15, 5))
plt.plot(new_day, label='New Day Traffic', marker='o')
plt.title('Traffic Analysis for New Day (DBSCAN Clustering)')
plt.xlabel('Hour')
plt.ylabel('Traffic')
plt.grid(True)
anomalies_dbscan = []
for i, val in enumerate(new_day):
    if dbscan.fit_predict(df_scaled[i].reshape(1, -1)) == -1:
        anomalies_dbscan.append(i)
        plt.scatter(i, val, color='red')
plt.legend()
plt.show()

# 输出结果
print("K-means Cluster for New Day:", new_day_cluster)
print("DBSCAN Cluster for New Day:", new_day_dbscan[0])
print("Anomalies (K-means):", anomalies_kmeans)
print("Anomalies (DBSCAN):", anomalies_dbscan)
print("Anomaly ratio (DBSCAN):", anomaly_ratio)


