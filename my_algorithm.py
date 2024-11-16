import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim

# *******************REWARD********************************
from reward_algorithm import *  # v1~v8
# v9: 作为类成员函数，这里reward定义为全局变量，以反映全局策略。见下myPPOAlgorithm类

# PPO compute advantage funciton
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

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
    def __init__(self, state_dim, hidden_dim, action_dim, num_hidden_layers=2, residual_strength=0.2):
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

# # v3 add multi-head self-attention
# class PolicyNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim, num_hidden_layers=2, residual_strength=0.2, num_heads=4):
#         super(PolicyNet, self).__init__()
#         self.residual_strength = residual_strength
#         self.num_heads = num_heads

#         # input layer
#         self.input_fc = nn.Linear(state_dim, hidden_dim)

#         # Add attention layer
#         self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

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
    
# v4 add CNN



class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, num_hidden_layers=2, residual_strength=0):
        super(ValueNet, self).__init__()
        self.residual_strength = residual_strength
        # input layer
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(state_dim, hidden_dim))
        # hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        # output layers
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            residual = x * self.residual_strength
            x = F.leaky_relu(layer(x))
            if (x.shape == residual.shape) and (i % 2):
                x = x + residual  # 添加残差

        return self.fc_out(x)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # 使用Kaiming正态分布初始化权重
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        # 初始化偏置为零
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class myPPOAlgorithm:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, nums_episodes, state_dim, actor_hidden_dim, critic_hidden_dim, actor_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, residual_strength, 
                device, num_actor_hidden_layers=None, num_critic_hidden_layers=None, actor_pretrained_model=None, critic_pretrained_model=None, isTrain=True):
        self.actor_dim = actor_dim
        self.actor = PolicyNet(state_dim, actor_hidden_dim, actor_dim, num_actor_hidden_layers, residual_strength).to(device)
        self.critic = ValueNet(state_dim, critic_hidden_dim, num_critic_hidden_layers, residual_strength).to(device)
        if (actor_pretrained_model is None):
            self.actor.apply(initialize_weights)
            self.critic.apply(initialize_weights)
            pass
        else:
            self.actor.load_state_dict(torch.load(actor_pretrained_model))
            self.critic.load_state_dict(torch.load(critic_pretrained_model))
            pass
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=nums_episodes, eta_min=actor_lr/20)
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=nums_episodes, eta_min=critic_lr/20)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.isTrain = isTrain
        # 与env相关的变量
        self.env_max_steps = 100
        self.is_obstacled = False
        self.arrival_reward_flag = False
        self.reward = 0

    def reset(self):
        self.is_obstacled = False
        self.arrival_reward_flag = False
    # 全局奖励
    def reward_total_9(self, dist, pre_dist, obstacle_contact, step):
        # 1. 鼓励机械臂向目标物体前进
        delta = (dist - pre_dist)
        self.reward -= delta * 500

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            self.reward -= 3
            self.is_obstacled = True

        # 3. 到达奖励
        if dist < 0.05 and step <= self.env_max_steps:
            self.reward += 100
            if self.is_obstacled:
                self.reward -= 50
        elif step >= self.env_max_steps:
            self.reward -= (dist - 0.05) * 100
            if self.is_obstacled:
                self.reward -= 50

        return self.reward
    
    # 局部奖励
    def reward_total_9_1(self, dist, pre_dist, obstacle_contact, step):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进
        delta = (dist - pre_dist)
        reward -= delta * 600

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= 5
            self.is_obstacled = True

        # 3. 到达奖励
        # if dist < 0.05 and step <= self.env_max_steps:
        #     reward += 100
        #     if self.is_obstacled:
        #         reward -= 30
        # elif step >= self.env_max_steps:
        #     reward -= (dist - 0.05) * 100
        #     if self.is_obstacled:
        #         reward -= 30
        if (not self.arrival_reward_flag) and (step >= self.env_max_steps or dist < 0.05):
            if dist < 0.05:
                # debug
                pass
            reward -= (dist - 0.1) * 100
            if self.is_obstacled:
                reward -= 30
            self.arrival_reward_flag = True

        return reward
    def reward_total_9_2(self, dist, pre_dist, obstacle_contact, step):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进
        delta = (dist - pre_dist)
        reward -= delta * 800

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= 8
            self.is_obstacled = True

        # 3. 到达奖励
        if (not self.arrival_reward_flag) and (step >= self.env_max_steps or dist < 0.05):
            reward -= (dist - 0.1) * 100
            if self.is_obstacled:
                reward -= 50
            self.arrival_reward_flag = True

        return reward

    def preprocess_state(self, state):
        return state

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
        

    def get_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        self.actor.eval()
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return np.array(action.squeeze(0).tolist())  # 6 维度的动作

        if self.isTrain and np.random.rand() < 0.2:
            random_action = np.random.uniform(-1, 1, size=action.shape[-1])
            return random_action
        else:
            return np.array(action.squeeze(0).tolist())  # 6 维度的动作

    def update(self, transition_dict):
        self.reward = 0
        self.actor.train()
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions).sum(dim=1, keepdim=True)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions).sum(dim=1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # lr decay
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        return actor_loss.item(), critic_loss.item()