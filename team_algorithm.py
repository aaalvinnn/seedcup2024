import numpy as np
from stable_baselines3 import PPO
from abc import ABC, abstractmethod
from my_algorithm import *

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

class MyCustomAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.state_dim = 12
        self.hidden_dim = 128
        self.action_dim = 6
        self.actor = PolicyNet(self.state_dim, self.hidden_dim, self.action_dim)
        self.actor.load_state_dict(torch.load('output/202411111405/actor_best.pth'))
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
    

