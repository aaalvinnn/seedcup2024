from stable_baselines3 import PPO
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import os
import numpy as np
from collections import deque

class BaseAlgorithm(ABC):
    @abstractmethod 
    def get_action(self, observation):
        """
        输入观测值，返回动作
        Args:
            observation: numpy array of shape (1, 12) 包含:
                - 6个关节角度 (归一化到[0,1])
                - 3个目标位置坐标 ()
                - 3个障碍物位置坐标 ()
        Returns:
            action: numpy array of shape (6,) 范围在[-1,1]之间
        """
        pass

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, num_hidden_layers, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.action_bound = action_bound
        # input layer
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(state_dim, hidden_dim))

        # hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))

        # output layers
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))

        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = torch.distributions.Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob

class MyCustomAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.state_dim = 12
        self.hidden_dim = 64
        self.num_hidden_layers = 6
        self.action_dim = 6
        self.actor = PolicyNet(self.state_dim, self.hidden_dim, self.num_hidden_layers, self.action_dim, 1)
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        self.actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.actor.eval()
        pass
    
    def get_action(self, state):
        state = np.concatenate([state[0]], axis=-1)
        state = torch.tensor([state], dtype=torch.float)
        action = self.actor(state)[0]
        return np.array(action.squeeze(0).tolist()) 

# 示例：使用PPO预训练模型
class PPOAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model = PPO.load("output/1203/1948_n_steps-3_score65/best_score_model.zip", device="cpu")
        # 多步观测
        self.n_state_steps = self.model.observation_space.shape[0]
        self.state_buffer = deque(maxlen=self.n_state_steps)
        self.reset_buffer()
        
    def reset_buffer(self):
        for _ in range(self.n_state_steps):
            # init_state = np.array([[0.36261892, 0.3400045 , 0.11557633, 0.04461199, 0.36262435, 0.49998325,
            #                        0.0110548 , 0.82783694, 0.18774934, 0.15527243, 0.6, 0.10094377]])
            init_state = np.array([[0.5 for i in range(12)]])
            self.state_buffer.append(init_state)    # use init state to fill buffer

    def get_action(self, observation):
        self.state_buffer.append(observation)
        state = np.concatenate(list(self.state_buffer), axis=0)
        action, _ = self.model.predict(state)
        return action
    

