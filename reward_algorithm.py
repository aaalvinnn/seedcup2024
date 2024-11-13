import numpy as np
import math

def reward_dist_1(dist):
    return 1 - (dist - 0.2)/0.15

def reward_dist_2(dist):
    return 1 - (dist - 0.2) * 8

def reward_dist_2_1(dist):
    return 1 - (dist - 0.3) * 10

def reward_dist_3(dist):
    """
    分段
    """
    if dist > 0.2:
        return 1 / (1 + 5 * (dist - 0.2))
    elif dist <= 0.2 and dist >= 0.05:
        return 1 / (1 + dist)
    else:
        print("ERROR")
        exit(-1)

def reward_dist_4(dist):
    """
    不分段，最大reward也只会是4（即dist=0.05）
    """
    return 1 / (1 + 5 * (dist - 0.2))

def reward_dist_5(dist):
    """
    幂函数
    """
    return 1 / math.pow(math.e, dist)

def reward_total_1(dist, obstacle_contact):
    # 碰到障碍物，之后奖励全为-3
    # 取-3是因为reward_2最差值也不会低于-3
    if obstacle_contact:
        return -3
    return reward_dist_2(dist)

def reward_total_2(dist, obstacle_contact):
    # 碰到障碍物，直接break掉
    if obstacle_contact:
        return -4
    return reward_dist_2(dist)

def reward_total_3(dist, obstacle_contact):
    # 碰到障碍物，这个step计算负奖励，不影响之后的step
    if obstacle_contact:
        return -3
    return reward_dist_2(dist)

def reward_total_4(dist, obstacle_contact):
    # 碰到障碍物，这个step计算负奖励，不影响之后的step
    if obstacle_contact:
        return -1
    return reward_dist_5(dist)

def reward_total_5(dist, obstacle_contact):     # release v1
    # 碰到障碍物，这个step计算负奖励，不影响之后的step
    if obstacle_contact:
        return -1 + reward_dist_2_1(dist)
    return reward_dist_2_1(dist)

def reward_total_5_1(dist, obstacle_contact):
    # 碰到障碍物，这个step计算负奖励，不影响之后的step
    if obstacle_contact:
        return -1 + reward_dist_4(dist)
    return reward_dist_4(dist)

def reward_total_6(dist, pre_dist, obstacle_contact):
    reward = 1 - (dist - 0.3) * 10
    # 考虑delta dist
    delta = (dist - pre_dist)
    reward -= delta * 20
    if obstacle_contact:
        reward -= 1

    return reward

def reward_total_7(dist, pre_dist, obstacle_contact):
    reward = 0
    if dist > 0.3:
        reward = 2 - (dist - 0.3) * 20
    else:
        reward = 1 / (0.2 + dist)
    # 考虑delta dist (about 0.005m)
    delta = (dist - pre_dist)
    reward -= delta * 40
    if obstacle_contact:
        reward -= 1

    return reward

def reward_total_7_1(dist, pre_dist, obstacle_contact):
    reward = 0
    if dist > 0.2:
        reward = 2 - (dist - 0.2) * 20
    else:
        reward = 1 / (0.3 + dist)
    # 考虑delta dist (about 0.005m)
    delta = (dist - pre_dist)
    reward -= delta * 40
    if obstacle_contact:
        reward -= 1

    return reward

def reward_total_8(dist, pre_dist, obstacle_contact):
    reward = 1 - (dist - 0.3) * 15

    # 考虑delta dist (about 0.005m)
    delta = (dist - pre_dist)
    reward -= delta * 50
    if obstacle_contact:
        reward -= 1

    return reward

def reward_total_8_1(dist, pre_dist, obstacle_contact):
    reward = 1 - (dist - 0.3) * 20

    # 考虑delta dist (about 0.005m)
    delta = (dist - pre_dist)
    reward -= delta * 50
    # if obstacle_contact:
    #     reward -= 1

    return reward
