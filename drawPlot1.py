import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# 生成正常流量数据
def generate_normal_flow_data(num_days, points_per_day, seed=42):
    np.random.seed(seed)
    normal_flow = []
    for day in range(num_days):
        daily_pattern = np.sin(np.linspace(0, 2 * np.pi, points_per_day)) * 10 + 20
        noise = np.random.normal(0, 2, points_per_day)
        daily_flow = daily_pattern + noise
        normal_flow.append(daily_flow)
    normal_flow = np.array(normal_flow).flatten()
    return normal_flow

# 生成异常流量数据
def generate_anomalous_flow_data(normal_flow, anomaly_ratio=0.05, anomaly_magnitude=15, seed=42):
    np.random.seed(seed)
    anomalous_flow = normal_flow.copy()
    num_anomalies = int(len(normal_flow) * anomaly_ratio)
    anomaly_indices = np.random.choice(len(normal_flow), num_anomalies, replace=False)
    for idx in anomaly_indices:
        anomalous_flow[idx] += anomaly_magnitude * np.random.choice([-1, 1])
    return anomalous_flow

# 生成合成数据
num_days = 30
points_per_day = 48  # 每天的时间点数（例如，每半小时一个点）
normal_flow = generate_normal_flow_data(num_days, points_per_day)
anomalous_flow = generate_anomalous_flow_data(normal_flow)

# 创建DataFrame保存数据
timestamps = pd.date_range(start='2023-01-01', periods=len(normal_flow), freq='30min')
data = pd.DataFrame({'timestamp': timestamps, 'normal_flow': normal_flow, 'anomalous_flow': anomalous_flow})

# 保存到CSV文件
data.to_csv('synthetic_traffic_flow_data.csv', index=False)
print("Synthetic data saved to 'synthetic_traffic_flow_data.csv'")

# 可视化合成数据
plt.figure(figsize=(15, 6))
plt.plot(data['timestamp'], data['normal_flow'], label='Normal Flow')
plt.plot(data['timestamp'], data['anomalous_flow'], label='Anomalous Flow', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Flow')
plt.title('Synthetic Traffic Flow Data')
plt.legend()
plt.show()