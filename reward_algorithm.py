import numpy as np

def reward_1(dist):
    return 1 - (dist - 0.2)/0.15

def reward_2(dist):
    return 1 - (dist - 0.2) * 8

def reward_3(dist, obstacle_contact):
    # 碰到障碍物，之后奖励全为-3
    if obstacle_contact:
        return -3
    return reward_2(dist)

def reward_4(dist, obstacle_contact):
    # 碰到障碍物，直接break掉
    if obstacle_contact:
        return -4
    return reward_2(dist)
