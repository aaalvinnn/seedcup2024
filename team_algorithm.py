from stable_baselines3 import PPO
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import os
import numpy as np

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

# v1
# class PolicyNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim, num_hidden_layers=2):
#         super(PolicyNet, self).__init__()
#         # input layer
#         self.layers = torch.nn.ModuleList()
#         self.layers.append(torch.nn.Linear(state_dim, hidden_dim))
#         # hidden layers
#         for _ in range(num_hidden_layers - 1):
#             self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
#         # output layers
#         self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
#         self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         for layer in self.layers:
#             x = F.relu(layer(x))
#         mu = torch.tanh(self.fc_mu(x))
#         std = F.softplus(self.fc_std(x))
#         return mu, std

# v2
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, num_hidden_layers=2, residual_strength=0):
        super(PolicyNet, self).__init__()
        self.residual_strength = residual_strength
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
            residual = self.residual_strength * x
            x = F.leaky_relu(layer(x))
            if (x.shape == residual.shape) and (i % 2):
                x = x + residual  # 添加残差

        mu = torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        # std = torch.clamp(std, min=1e-6)
        return mu, std

# v3 add multi-head attention
# v3 add multi-head attention
# class PolicyNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim, num_hidden_layers=2, residual_strength=0.2, num_heads=4):
#         super(PolicyNet, self).__init__()
#         self.residual_strength = residual_strength
#         self.num_heads = num_heads

#         # input layer
#         self.input_fc = torch.nn.Linear(state_dim, hidden_dim)

#         # Add attention layer
#         self.attention = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

#         # hidden layers
#         self.layers = torch.nn.ModuleList()
#         for _ in range(num_hidden_layers):
#             self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))

#         # output layers
#         self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
#         self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         x = self.input_fc(x)
#         x, _ = self.attention(x, x, x)
#         x = F.relu(x)

#         for i, layer in enumerate(self.layers):
#             residual = self.residual_strength * x
#             x = F.relu(layer(x))
#             if (x.shape == residual.shape) and (i % 2):
#                 x = x + residual  # 添加残差

#         mu = torch.tanh(self.fc_mu(x))
#         std = F.softplus(self.fc_std(x))
#         # std = torch.clamp(std, min=1e-6)
#         return mu, std
    
class MyCustomAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.state_dim = 12
        self.hidden_dim = 128
        self.action_dim = 6
        self.actor = PolicyNet(self.state_dim, self.hidden_dim, self.action_dim, 5, 0)
        model_path = os.path.join(os.path.dirname(__file__), "model_best.pth")
        self.actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        pass
    
    def preprocess_state(self, state):
        jixiebi_state = state[:6]
        dest_stateX, dest_stateY, dest_stateZ = state[6:9]
        obstacle_stateX, obstacle_stateY, obstacle_stateZ = state[9:]

        dest_stateX = (dest_stateX - (-0.2)) / (0.2 - (-0.2))
        dest_stateY = (dest_stateY - (0.8)) / (0.9 - 0.8)
        dest_stateZ = (dest_stateZ - (0.1)) / (0.3 - 0.1)

        obstacle_stateX = (obstacle_stateX - (-0.4)) / (0.4 - (-0.4))
        obstacle_stateY = obstacle_stateY
        obstacle_stateZ = (obstacle_stateZ - (0.1)) / (0.3 - 0.1)

        return np.concatenate([jixiebi_state, [dest_stateX], [dest_stateY], [dest_stateZ], [obstacle_stateX], [obstacle_stateY], [obstacle_stateZ]])
    
    def get_action(self, observation):
        self.actor.eval()
        state = torch.tensor([observation[0]], dtype=torch.float)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action.squeeze(0).tolist()

# 示例：使用PPO预训练模型
class PPOAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model = PPO.load("ppo_stablebaselines3_env.zip", device="cpu")

    def get_action(self, observation):
        action, _ = self.model.predict(observation)
        return action
    

