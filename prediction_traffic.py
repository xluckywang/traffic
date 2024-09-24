import numpy as np
import pandas as pd
import random
from collections import deque
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['timestamp'])
    data = data.fillna(0)
    print("Data columns:", data.columns)  # 打印列名用于调试
    features = data[['anomalous_flow']].values  # 只选择anomalous_flow特征
    return features

data = load_and_preprocess_data('synthetic_traffic_flow_data.csv')

historical_data = np.random.randint(10, 50, size=(30, 48, 1))  # 示例历史数据

threshold = 30

def compute_normality_score(flow_value, reference_flows, bandwidth=1.0):
    if reference_flows.shape[0] == 0:
        return 1.0
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(reference_flows.reshape(-1, 1))
    log_density = kde.score_samples(flow_value.reshape(-1, 1))
    density = np.exp(log_density)
    return density[0]

def get_reference_flows(historical_data, current_time_index, window_size=2):
    reference_flows = []
    for day in historical_data:
        start_idx = max(0, current_time_index - window_size)
        end_idx = min(day.shape[0], current_time_index + window_size + 1)
        reference_flows.extend(day[start_idx:end_idx, 0])
    return np.array(reference_flows)

def compute_reward(normality_score, action):
    if normality_score < 1:
        if action == 1:
            return 1 / normality_score
        else:
            return -1 / normality_score
    else:
        if action == 0:
            return normality_score
        else:
            return -normality_score

class LSTMDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMDQN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 128).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 128).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

input_dim = 1
hidden_dim = 128
output_dim = 2

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 5e-6
        self.model = LSTMDQN(state_size, 128, action_size).to(device)
        self.target_model = LSTMDQN(state_size, 128, action_size).to(device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            target = self.model(state)
            if done:
                target[0][action] = reward
            else:
                with torch.no_grad():
                    t = self.target_model(next_state)
                target[0][action] = reward + self.gamma * torch.max(t[0]).item()
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def compute_reward(self, flow_value, reference_flows, action):
        normality_score = compute_normality_score(flow_value, reference_flows)
        reward = compute_reward(normality_score, action)
        return reward

def train_dqn(agent, data, historical_data, episodes=1000, batch_size=32):
    precision_list, recall_list, f1_list = [], [], []
    for e in tqdm(range(episodes)):
        state = data[0:20, :]  # 确保形状为(20, 1)
        true_labels = []
        pred_labels = []
        for time in range(20, len(data)):
            action = agent.act(state)
            next_state = data[time-19:time+1, :]  # 确保形状为(20, 1)
            reference_flows = get_reference_flows(historical_data, time)
            reward = agent.compute_reward(data[time], reference_flows, action)
            done = (time == len(data)-1)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            true_labels.append(1 if data[time][0] > threshold else 0)
            pred_labels.append(action)
            if done:
                agent.update_target_model()
                precision = precision_score(true_labels, pred_labels, zero_division=1)
                recall = recall_score(true_labels, pred_labels, zero_division=1)
                f1 = f1_score(true_labels, pred_labels, zero_division=1)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                print(f"episode: {e}/{episodes}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, e: {agent.epsilon:.4f}")
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    plt.figure(figsize=(12, 6))
    plt.plot(range(episodes), precision_list, label='Precision')
    plt.plot(range(episodes), recall_list, label='Recall')
    plt.plot(range(episodes), f1_list, label='F1 Score')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

def main():
    data = load_and_preprocess_data('synthetic_traffic_flow_data.csv')
    historical_data = np.random.randint(10, 50, size=(30, 48, 1))

    agent = DQNAgent(state_size=1, action_size=2)

    train_dqn(agent, data, historical_data)

if __name__ == "__main__":
    main()
