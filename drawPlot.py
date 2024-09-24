import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# 生成示例数据
def generate_example_data(start_time, num_records, interval_minutes):
    timestamps = [start_time + pd.Timedelta(minutes=i*interval_minutes) for i in range(num_records)]
    flow = np.random.randint(10, 50, size=num_records)
    other_feature1 = np.random.uniform(5, 6, size=num_records)
    other_feature2 = np.random.uniform(3, 4, size=num_records)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'flow': flow,
        'other_feature1': other_feature1,
        'other_feature2': other_feature2
    })

    return data

start_time = pd.Timestamp('2023-01-01 00:00:00')
num_records = 1000
interval_minutes = 3

example_data = generate_example_data(start_time, num_records, interval_minutes)

# 假设历史数据是已经加载并处理好的多天的数据
historical_data = np.random.randint(10, 50, size=(30, 480))  # 示例历史数据

current_time_index = 150  # 当前时间步索引
window_size = 2  # 时间窗口大小

# 提取当前时间步和历史时间段的流量数据
current_flow = example_data.iloc[current_time_index]['flow']
reference_flows = []
for day in historical_data:
    start_idx = max(0, current_time_index - window_size)
    end_idx = min(len(day), current_time_index + window_size + 1)
    reference_flows.extend(day[start_idx:end_idx])
reference_flows = np.array(reference_flows)

plt.figure(figsize=(10, 6))
plt.plot(example_data['timestamp'], example_data['flow'], label='Current Day Flow')
plt.axvline(example_data['timestamp'].iloc[current_time_index], color='r', linestyle='--', label='Current Time Step')
plt.xlabel('Time')
plt.ylabel('Flow')
plt.title('Data Extraction (Step 1)')
plt.legend()
plt.show()


# 使用核密度估计进行聚类
kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(reference_flows.reshape(-1, 1))
log_density = kde.score_samples(reference_flows.reshape(-1, 1))
density = np.exp(log_density)

plt.figure(figsize=(10, 6))
plt.hist(reference_flows, bins=30, density=True, alpha=0.5, label='Historical Flows')
plt.axvline(current_flow, color='r', linestyle='--', label='Current Flow')
plt.plot(reference_flows, density, label='KDE Density')
plt.xlabel('Flow')
plt.ylabel('Density')
plt.title('Clustering (Step 2)')
plt.legend()
plt.show()


# 计算当前流量点的密度
current_density = kde.score_samples(np.array([[current_flow]]))
normality_score = np.exp(current_density)[0]

plt.figure(figsize=(10, 6))
plt.hist(reference_flows, bins=30, density=True, alpha=0.5, label='Historical Flows')
plt.axvline(current_flow, color='r', linestyle='--', label='Current Flow')
plt.plot(reference_flows, density, label='KDE Density')
plt.scatter([current_flow], [normality_score], color='red', zorder=5)
plt.text(current_flow, normality_score, f' {normality_score:.2f}', verticalalignment='bottom')
plt.xlabel('Flow')
plt.ylabel('Density')
plt.title('Normality Score Calculation (Step 3)')
plt.legend()
plt.show()


def compute_reward(normality_score, action):
    if normality_score < 1:
        if action == 1:
            return 1 / normality_score  # 正确检测到异常
        else:
            return -1 / normality_score  # 未检测到异常
    else:
        if action == 0:
            return normality_score  # 正确检测到正常
        else:
            return -normality_score  # 错误检测为异常

action = 1  # 假设采取的动作
reward = compute_reward(normality_score, action)

plt.figure(figsize=(10, 6))
plt.hist(reference_flows, bins=30, density=True, alpha=0.5, label='Historical Flows')
plt.axvline(current_flow, color='r', linestyle='--', label='Current Flow')
plt.plot(reference_flows, density, label='KDE Density')
plt.scatter([current_flow], [normality_score], color='red', zorder=5)
plt.text(current_flow, normality_score, f' {normality_score:.2f}', verticalalignment='bottom')
plt.xlabel('Flow')
plt.ylabel('Density')
plt.title(f'Reward Computation (Step 4), Reward: {reward:.2f}')
plt.legend()
plt.show()
