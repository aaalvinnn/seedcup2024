import os
import numpy as np
import pybullet as p
import pybullet_data
import math
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import torch

class myTrainingEnv(gym.Env):
    def __init__(self,is_senior,seed, nums_episodes, gui=False):
        super(myTrainingEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 12), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6, ), dtype=np.float32)
        self.seed = seed
        self.is_senior = is_senior
        self.step_num = 0
        self.max_steps = 100
        self.pre_dist = 0
        self.p = bullet_client.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self.p.setGravity(0, 0, -9.81)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.obstacles = 0
        self.max_epochs = nums_episodes
        self.epoch = 0
        self.init_env()

    def init_env(self):
        np.random.seed(self.seed)  
        self.fr5 = self.p.loadURDF("fr5_description/urdf/fr5v6.urdf", useFixedBase=True, basePosition=[0, 0, 0],
                                   baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi]), flags=p.URDF_USE_SELF_COLLISION)
        self.table = self.p.loadURDF("table/table.urdf", basePosition=[0, 0.5, -0.63],
                                      baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        collision_target_id = self.p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=0.02, height=0.05)
        self.target = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_target_id, basePosition=[0.5, 0.8, 2])
        collision_obstacle_id = self.p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=0.1)
        self.obstacle1 = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_obstacle_id, basePosition=[0.5, 0.5, 2])
        self.reset()

    def reset(self, seed=None,  **kwargs):
        self.step_num = 0
        self.success_reward = 0
        self.training_reward = 0
        self.obstacles = 0
        self.epoch += 1
        self.terminated = False
        self.obstacle_contact = False
        neutral_angle = [-49.45849125928217, -57.601209583849, -138.394013961943, -164.0052115563118, -49.45849125928217, 0, 0, 0]
        neutral_angle = [x * math.pi / 180 for x in neutral_angle]
        self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL, targetPositions=neutral_angle)

        self.goalx = np.random.uniform(-0.2, 0.2, 1)
        self.goaly = np.random.uniform(0.8, 0.9, 1)
        self.goalz = np.random.uniform(0.1, 0.3, 1)
        self.target_position = [self.goalx[0], self.goaly[0], self.goalz[0]]
        self.p.resetBasePositionAndOrientation(self.target, self.target_position, [0, 0, 0, 1])

        self.obstacle1_position = [np.random.uniform(-0.2, 0.2, 1) + self.goalx[0], 0.6, np.random.uniform(0.1, 0.3, 1)]
        self.p.resetBasePositionAndOrientation(self.obstacle1, self.obstacle1_position, [0, 0, 0, 1])
        for _ in range(100):
            self.p.stepSimulation()

        self.pre_dist = self.get_dis()
        
        return self.get_observation(), {}

    def get_observation(self):
        joint_angles = [self.p.getJointState(self.fr5, i)[0] * 180 / np.pi for i in range(1, 7)]
        obs_joint_angles = ((np.array(joint_angles, dtype=np.float32) / 180) + 1) / 2
        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])
        obstacle1_position = np.array(self.p.getBasePositionAndOrientation(self.obstacle1)[0])
        self.observation = np.hstack((obs_joint_angles, target_position, obstacle1_position)).flatten().reshape(1, -1)
        return self.observation

    def step(self, action):
        if self.terminated:
            return self.reset_episode()
        
        self.step_num += 1
        joint_angles = [self.p.getJointState(self.fr5, i)[0] for i in range(1, 7)]
        action = np.clip(action, -1, 1)
        fr5_joint_angles = np.array(joint_angles) + (np.array(action[:6]) / 180 * np.pi)
        gripper = np.array([0, 0])
        angle_now = np.hstack([fr5_joint_angles, gripper])
        self.reward()
        self.p.setJointMotorControlArray(self.fr5, [1, 2, 3, 4, 5, 6, 8, 9], p.POSITION_CONTROL, targetPositions=angle_now)

        for _ in range(20):
            self.p.stepSimulation()

        observation = self.get_observation()
        reward = self.get_reward()
        done = self.terminated

        return observation, reward, done, False, {}

    def get_dis(self):
        gripper_pos = self.p.getLinkState(self.fr5, 6)[0]
        relative_position = np.array([0, 0, 0.15])
        rotation = R.from_quat(self.p.getLinkState(self.fr5, 7)[1])
        rotated_relative_position = rotation.apply(relative_position)
        gripper_centre_pos = np.array(gripper_pos) + rotated_relative_position
        target_position = np.array(self.p.getBasePositionAndOrientation(self.target)[0])
        return np.linalg.norm(gripper_centre_pos - target_position)

    def reward(self):
        # 获取与桌子和障碍物的接触点
        table_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.table)
        obstacle1_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.obstacle1)

        for contact_point in table_contact_points or obstacle1_contact_points:
            link_index = contact_point[3]
            if link_index not in [0, 1]:
                self.obstacle_contact = True

        # 计算奖励
        if self.get_dis() < 0.05 and self.step_num <= self.max_steps:
            self.success_reward = 100
            if self.obstacle_contact:
                if self.is_senior:
                    self.success_reward = 20
                elif not self.is_senior:
                    self.success_reward = 50
                else:
                    return 
            self.terminated = True

        elif self.step_num >= self.max_steps:
            distance = self.get_dis()
            if 0.05 <= distance <= 0.2:
                self.success_reward = 100 * (1 - ((distance - 0.05) / 0.15))
            else:
                self.success_reward = 0
            if self.obstacle_contact:
                if self.is_senior:
                    self.success_reward *= 0.2 
                elif not self.is_senior:
                    self.success_reward *= 0.5
                    
            self.terminated = True


    def reset_episode(self):
        self.reset()
        return self.step_num, self.get_dis()

    def close(self):
        self.p.disconnect()

    # 以下为暴露给train.py的接口
    def is_obstacle_contact(self):
        """
        判断是否接触到障碍物，在不修改env.py的情况下，作为接口供train.py使用
        """
        # 获取与桌子和障碍物的接触点
        table_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.table)
        obstacle1_contact_points = self.p.getContactPoints(bodyA=self.fr5, bodyB=self.obstacle1)

        for contact_point in table_contact_points or obstacle1_contact_points:
            link_index = contact_point[3]
            if link_index not in [0, 1]:
                self.obstacles += 1
                return True
            
        return False
    
    def get_step_now(self):
        return self.step_num

    # def get_reward(self):
    #     reward = 0
    #     dist = self.get_dis()
    #     # 1. 鼓励机械臂向目标物体前进
    #     delta = (dist - self.pre_dist)
    #     reward -= delta * 500
    #     self.pre_dist = dist

    #     # 2. 移动时碰到障碍物
    #     if self.is_obstacle_contact():
    #         reward -= 5

    #     # 3. 到达奖励
    #     if dist < 0.05 and self.step_num <= self.max_steps:
    #         reward += 100
    #         if self.obstacle_contact:
    #             reward -= 50
    #     elif self.step_num >= self.max_steps:
    #         reward -= (dist - 0.05) * 100
    #         if self.obstacle_contact:
    #             reward -= 50

    #     return reward
    def get_reward(self):
        reward = 0
        dist = self.get_dis()
        # 1. 鼓励机械臂向目标物体前进
        delta = (dist - self.pre_dist)
        reward -= delta * 800
        self.pre_dist = dist

        # 2. 移动时碰到障碍物
        if self.is_obstacle_contact():
            reward -= 6 + self.obstacles * 0.15

        # 3. 余弦衰减
        if self.epoch >= self.max_epochs / 2:
            reward *= (1 + math.cos(math.pi * (self.epoch-self.max_epochs / 2) / self.max_epochs)) / 2

        # 4. 到达奖励
        if dist < 0.05 or self.step_num <= self.max_steps:
            reward += self.success_reward

        return reward

def train(env: myTrainingEnv, nums_episodes):
    # 创建 PPO 模型
    model = PPO(policy="MlpPolicy", env=env, learning_rate=1e-5, verbose=1, device="cuda")
    model = model.load("model.zip", env, device="cuda")

    # 开始训练
    model.learn(total_timesteps=nums_episodes*100)

    # 保存模型
    model.save("ppo_stablebaselines3_env")

def test(env: myTrainingEnv):
    # 加载模型并测试
    model = PPO.load("ppo_stablebaselines3_env")
    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

def model_output(output_path):
    model = PPO.load("ppo_stablebaselines3_env")
    policy_net = model.policy
    torch.save(policy_net.state_dict(), os.path.join(output_path, "model.pth"))

if __name__ == "__main__":
    nums_episodes = 1200
    env = myTrainingEnv(is_senior=True, seed=100, nums_episodes=nums_episodes)
    train(env, nums_episodes)
    # test(env)
    # model_output("pretrained_models/PPO-13")


