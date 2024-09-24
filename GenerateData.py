import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 生成示例数据
def generate_example_data(start_time, num_records, interval_minutes):
    timestamps = [start_time + timedelta(minutes=i*interval_minutes) for i in range(num_records)]
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

start_time = datetime(2023, 1, 1, 0, 0, 0)
num_records = 1000
interval_minutes = 3

example_data = generate_example_data(start_time, num_records, interval_minutes)
example_data.to_csv('path_to_traffic_flow_data.csv', index=False)
print("示例数据已生成并保存到 'path_to_traffic_flow_data.csv'")
