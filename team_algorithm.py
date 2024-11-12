from stable_baselines3 import PPO
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import os

class BaseAlgorithm(ABC):
    @abstractmethod 
    def get_action(self, observation):
        """
        输入观测值，返回动作
        Args:
            observation: numpy array of shape (1, 12) 包含:
                - 6个关节角度 (归一化到[0,1])
                - 3个目标位置坐标
                - 3个障碍物位置坐标
        Returns:
            action: numpy array of shape (6,) 范围在[-1,1]之间
        """
        pass

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, num_hidden_layers=2):
        super(PolicyNet, self).__init__()
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
        for layer in self.layers:
            x = F.relu(layer(x))
        mu = torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
class MyCustomAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.state_dim = 12
        self.hidden_dim = 64
        self.action_dim = 6
        self.actor = PolicyNet(self.state_dim, self.hidden_dim, self.action_dim, 4)
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        self.actor.load_state_dict(torch.load(model_path))
        pass
        
    def get_action(self, observation):
        state = torch.tensor([observation[0]], dtype=torch.float)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return action.squeeze(0).tolist()

# 示例：使用PPO预训练模型
class PPOAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model = PPO.load("model.zip", device="cpu")

    def get_action(self, observation):
        action, _ = self.model.predict(observation)
        return action
    

