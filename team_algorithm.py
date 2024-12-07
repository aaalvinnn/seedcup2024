from stable_baselines3 import PPO
from abc import ABC, abstractmethod
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

class MyCustomAlgorithm(BaseAlgorithm):
    def __init__(self):
        self.model = PPO.load("output/1206/1659_n_steps-5_score90/model.zip", device="cpu")
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
    

