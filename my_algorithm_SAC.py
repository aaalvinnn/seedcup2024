import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy

# v2
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

# v1
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, num_hidden_layers, action_dim):
        super(ValueNet, self).__init__()
        # input layer
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(state_dim + action_dim, hidden_dim))
        # hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        # output layers
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))

        return self.fc_out(x)
    
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # 使用Kaiming正态分布初始化权重
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # 初始化偏置为零
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
class mySACAlgorithm:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, nums_episodes, state_dim, actor_hidden_dim, critic_hidden_dim, action_dim, num_actor_hidden_layers, num_critic_hidden_layers, actor_lr, critic_lr, alpha_lr, 
                 target_entropy, tau, gamma, device, pretrained_actor=None, pretrained_critic_1=None, pretrained_critic_2=None, a=800, b=6, c=0.1, d=5, e=1):
        self.action_bound = 1
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, actor_hidden_dim, num_actor_hidden_layers, action_dim, self.action_bound).to(device)  # 策略网络
        self.critic_1 = ValueNet(state_dim, critic_hidden_dim, num_critic_hidden_layers,action_dim).to(device)  # 第一个Q网络
        self.critic_2 = ValueNet(state_dim, critic_hidden_dim, num_critic_hidden_layers,action_dim).to(device)  # 第二个Q网络
        if (pretrained_actor == None):
            self.actor.apply(initialize_weights)
            self.critic_1.apply(initialize_weights)
            self.critic_2.apply(initialize_weights)
        else:
            self.actor.load_state_dict(torch.load(pretrained_actor, weights_only=True))
            self.critic_1.load_state_dict(torch.load(pretrained_critic_1, weights_only=True))
            self.critic_2.load_state_dict(torch.load(pretrained_critic_2, weights_only=True))
        self.target_critic_1 = ValueNet(state_dim, critic_hidden_dim, num_critic_hidden_layers,action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = ValueNet(state_dim, critic_hidden_dim, num_critic_hidden_layers,action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=nums_episodes, eta_min=actor_lr)
        # self.critic_1_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_1_optimizer, T_max=nums_episodes, eta_min=critic_lr)
        # self.critic_2_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.critic_2_optimizer, T_max=nums_episodes, eta_min=critic_lr)
        self.actor_scheduler = optim.lr_scheduler.OneCycleLR(
            self.actor_optimizer,
            max_lr=actor_lr*10,
            total_steps=nums_episodes*100,
            anneal_strategy='cos',
            cycle_momentum=True,
            div_factor=10,    # 学习率衰减的分割因子
        )
        self.critic_1_scheduler = optim.lr_scheduler.OneCycleLR(
            self.critic_1_optimizer,
            max_lr=critic_lr*10,
            total_steps=nums_episodes*100,
            anneal_strategy='cos',
            cycle_momentum=True,
            div_factor=10,    # 学习率衰减的分割因子
        )
        self.critic_2_scheduler = optim.lr_scheduler.OneCycleLR(
            self.critic_2_optimizer,
            max_lr=critic_lr*10,
            total_steps=nums_episodes*100,
            anneal_strategy='cos',
            cycle_momentum=True,
            div_factor=10,    # 学习率衰减的分割因子
        )
        
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.env_max_steps = 100
        self.a = a if a is not None else 800
        self.b = b if b is not None else 6
        self.c = c if c is not None else 0.1
        self.d = d if d is not None else 5
        self.e = e if e is not None else 1

    def reset(self, init_state):
        self.is_obstacled = False
        self.reward = 0
        self.pre_reward = 0

    def reward_total_10_3(self, dist, pre_dist, obstacle_contact, step, final_score):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进
        delta = (dist - pre_dist)
        reward -= delta * 800

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= 8
        
        # 3. 添加势能函数（与1. 不同，1. 中如果每次移动距离相同，则reward也是相同的；而势能函数是越靠近目标，reward越大）
        reward += (0.05 - dist) * 10

        # 4. 到达奖励
        if step >= self.env_max_steps or dist < 0.05:
            reward += final_score

        return reward

    
    def reward_total_10_3_1(self, dist, pre_dist, obstacle_contact, step, final_score):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进
        delta = (dist - pre_dist)
        reward -= delta * 800

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= 8
        
        # 3. 添加势能函数（与1. 不同，1. 中如果每次移动距离相同，则reward也是相同的；而势能函数是越靠近目标，reward越大）
        reward += (0.05 - dist) * 10

        # 4. 到达奖励
        reward /= 2
        if step >= self.env_max_steps or dist < 0.05:
            reward += final_score

        return reward
    
    def reward_total_10_3_3(self, dist, pre_dist, obstacle_contact, step, final_score):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进，1个step最大变化0.005m，dist的范围为0.05~1m
        # 范围 (-4, 4)
        delta = (dist - pre_dist)
        reward -= delta * 800

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= 8
        
        # 3. 添加势能函数，取值范围(-5, 5)
        reward += (0.05 - dist) * 10 + 5

        # 4. 到达奖励, 范围为0~100
        if step >= self.env_max_steps or dist < 0.05:
            reward += final_score

        return reward
    
    def reward_total_11_1(self, dist, pre_dist, obstacle_contact, n_obstacle, step, final_score):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进，1个step最大变化0.005m，dist的范围为0.05~1m
        # 范围 (-5, 5)
        delta = (dist - pre_dist)
        reward -= delta * 1000

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= 5
        
        # 3. 添加势能函数，取值范围(-5, 5)
        reward += (0.05 - dist) * 10 + 5

        # 4. 到达奖励, 范围为0~100
        if step >= self.env_max_steps or dist < 0.05:
            reward += final_score

        return reward
    
    def reward_total_11_2(self, dist, pre_dist, obstacle_contact, n_obstacle, step, final_score):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进，1个step最大变化0.005m，dist的范围为0.05~1m
        # 范围 (-5, 5)
        delta = (dist - pre_dist)
        reward -= delta * 1000

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= 5 + n_obstacle * 0.5
        
        # 3. 添加势能函数，取值范围(-5, 5)
        reward += (0.05 - dist) * 5 + 2.5

        # 4. 到达奖励, 范围为0~100
        if step >= self.env_max_steps or dist < 0.05:
            reward += final_score

        return reward
    
    def reward_total_11_3(self, dist, pre_dist, obstacle_contact, n_obstacle, step, final_score):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进，1个step最大变化0.005m，dist的范围为0.05~1m
        # 范围 (-5, 5)
        delta = (dist - pre_dist)
        reward -= delta * 1000

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= 8 + n_obstacle * 0.2

        # 3. 到达奖励, 范围为0~100
        if step >= self.env_max_steps or dist < 0.05:
            reward += final_score

        return reward
    
    def reward_total_11_4(self, dist, pre_dist, obstacle_contact, n_obstacle, step, final_score):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进，1个step最大变化0.005m，dist的范围为0.05~1m
        # 范围 (-5, 5)
        delta = (dist - pre_dist)
        reward -= delta * 800

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= 6 + n_obstacle * 0.2

        # 3. 到达奖励, 范围为0~100
        if step >= self.env_max_steps or dist < 0.05:
            reward += final_score

        return reward
    
    def reward_total_11_5(self, dist, pre_dist, obstacle_contact, n_obstacle, step, final_score):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进，1个step最大变化0.005m，dist的范围为0.05~1m
        # 范围 (-5, 5)
        delta = (dist - pre_dist)
        reward -= delta * 800

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= 6 + n_obstacle * 0.1

        # 3. 添加势能函数，取值范围(-5, 5)
        reward += (0.05 - dist) * 5 + 2.5

        # 4. 到达奖励, 范围为0~100
        if step >= self.env_max_steps or dist < 0.05:
            reward += final_score

        return reward
    
    def reward_total_11_5_test(self, dist, pre_dist, obstacle_contact, n_obstacle, step, final_score):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进，1个step最大变化0.005m，dist的范围为0.05~1m
        # 范围 (-5, 5)
        delta = (dist - pre_dist)
        reward -= delta * self.a

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= self.b + n_obstacle * self.c

        # 3. 添加势能函数，取值范围(-5, 5)
        reward += (0.55 - dist) * self.d

        # 4. 衰减reward shaping
        reward *= self.e

        # 5. 到达奖励, 范围为0~100
        if step >= self.env_max_steps or dist < 0.05:
            reward += final_score

        return reward
    
    def reward_total_12_test(self, dist, pre_dist, obstacle_contact, n_obstacle, step, final_score):
        reward = 0
        # 1. 鼓励机械臂向目标物体前进，1个step最大变化0.005m，dist的范围为0.05~1m
        # 范围 (-5, 5)
        delta = (dist - pre_dist)
        reward -= delta * self.a

        # 2. 移动时碰到障碍物
        if obstacle_contact:
            reward -= self.b + n_obstacle * self.c

        # 3. 添加势能函数，取值范围(0, 20)，非线性
        reward += 1 / dist * self.d


        # 4. 衰减reward shaping
        reward *= self.e

        # 5. 到达奖励, 范围为0~100
        if step >= self.env_max_steps or dist < 0.05:
            reward += final_score

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

    def get_action(self, state, epsilon=0.1):
        # if np.random.rand() < epsilon:
        #     random_action = np.random.uniform(-1, 1, size=self.action_dim)
        #     return random_action
        state = np.concatenate([state], axis=-1)
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return np.array(action.squeeze(0).tolist()) 

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob.sum(dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict["next_states"],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        self.actor_scheduler.step()
        self.critic_1_scheduler.step()
        self.critic_2_scheduler.step()