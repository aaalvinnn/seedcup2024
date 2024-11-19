from env import Env
from my_algorithm import myPPOAlgorithm
from my_algorithm_SAC import mySACAlgorithm
from team_algorithm import PPOAlgorithm, MyCustomAlgorithm
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import os
import sys
from tqdm import tqdm
import json
import argparse
import collections
import reward_algorithm
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)
    
def env_step_log(env: Env, action, reward, obstacle_contact):
    print(f"Step: {env.step_num}, dest_dist: {env.get_dis()}, is_Obstacle: {obstacle_contact}, Reward: {reward}")
        # State: {env.get_observation()[0]}\nAction: {action}\n")

def train_offline_policy_agent(algorithm: mySACAlgorithm, num_episodes, replay_buffer_size, minimal_size, batch_size, config, is_log):
    replay_buffer = ReplayBuffer(replay_buffer_size)
    
    if is_log:
        cur_time = datetime.now()
        output_fir_name = os.path.join("output", cur_time.strftime("%m%d"), cur_time.strftime("%H%M") + "_" + config)
        output_dir = os.path.join(os.path.dirname(__file__), output_fir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        writer = SummaryWriter(os.path.join(output_dir, 'SummaryWriter'))
        log_file = open(os.path.join(output_dir, 'log.txt'), 'w+')
        sys.stdout = log_file

    env = Env(is_senior=True,seed=100,gui=False)
    done = False
    total_score_list = []
    score50_best = 0
    total_reward_list = []
    total_obstacle_list = []
    total_dist_list = []

    for i in range(int(num_episodes/50)):
        with tqdm(total=50, desc='Iteration %d' % i) as pbar:
            for i_episode in range(50):
                algorithm.reset(env.reset()[0])
                epoch = 50 * i + i_episode + 1
                score = 0
                total_reward = 0
                total_obstacle = 0
                done = False
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = algorithm.preprocess_state(env.reset()[0])
                pre_dist = env.get_dis()

                print(f"******** Episode:", epoch, "*******")
                while not done:
                    action = algorithm.get_action(state)
                    _ = env.step(action)
                    new_state = algorithm.preprocess_state(env.get_observation()[0])
                    done = env.terminated
                    # reward = reward_algorithm.reward_total_8_1_1(env.get_dis(), pre_dist, env.is_obstacle_contact(), env.get_step_now())
                    reward = algorithm.reward_total_10_3_3(env.get_dis(), pre_dist, env.is_obstacle_contact(), env.get_step_now(), env.get_score())
                    replay_buffer.add(state, action, reward, new_state, done)
                    total_reward += reward
                    score += env.success_reward
                    pre_dist = env.get_dis()
                    state = new_state
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        algorithm.update(transition_dict)

                    if is_log:
                        env_step_log(env, action, reward, env.is_obstacle_contact())

                    if (env.is_obstacle_contact()):
                        total_obstacle += 1
                        # break

                total_reward_list.append(total_reward)
                total_score_list.append(score)
                total_obstacle_list.append(total_obstacle)
                total_dist_list.append(env.get_dis())
                print(f"Train_{epoch} completed. steps:", env.step_num, "Distance:", env.get_dis(), "Score:", score, "Reward:", total_reward, "n_Obstacle: ", total_obstacle)

                # Tensorboard logging
                if is_log:
                    writer.add_scalar('Total Reward', total_reward, epoch)
                    writer.add_scalar('Score', score, epoch)
                    writer.add_scalar('End Distance', env.get_dis(), epoch)
                    writer.add_scalar('Obstacle Contact Num', total_obstacle, epoch)

                    # model saving
                    # torch.save(algorithm.actor.state_dict(), os.path.join(os.path.dirname(__file__), 'model.pth'))  # 放个在工程根目录下方便test.py测试
                    torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor.pth'))
                    torch.save(algorithm.critic_1.state_dict(), os.path.join(output_dir, 'critic_1.pth'))
                    torch.save(algorithm.critic_2.state_dict(), os.path.join(output_dir, 'critic_2.pth'))

                    if (epoch>=50):
                        score50_best = np.mean(total_score_list[-50:]) if np.mean(total_score_list[-50:]) > score50_best else score50_best
                        if (score50_best <= np.mean(total_score_list[-50:])):
                            torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor_best.pth'))
                            torch.save(algorithm.critic_1.state_dict(), os.path.join(output_dir, 'critic_1_best.pth'))
                            torch.save(algorithm.critic_2.state_dict(), os.path.join(output_dir, 'critic_2_best.pth'))
                            # torch.save(algorithm.actor.state_dict(), os.path.join(os.path.dirname(__file__), 'model_best.pth'))  # 放个在工程根目录下方便test.py测试
                    if (epoch == 1):
                        torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor_init.pth'))    # 由于随机性，保存一下初始权重，提供预训练模型
                        torch.save(algorithm.critic_1.state_dict(), os.path.join(output_dir, 'critic_1_init.pth'))
                        torch.save(algorithm.critic_2.state_dict(), os.path.join(output_dir, 'critic_2_init.pth'))

                    display_reward = total_reward_list[-1]
                    display_score = total_score_list[-1]
                    display_obstacle = total_obstacle_list[-1]
                    display_dist = env.get_dis()
                    if (epoch >= 50):
                        display_reward = np.mean(total_reward_list[-50:])
                        display_score = np.mean(total_score_list[-50:])
                        display_obstacle = np.mean(total_obstacle_list[-50:])
                        display_dist = np.mean(total_dist_list[-50:])
                    pbar.set_postfix({
                        'episode':
                        '%d' % (epoch),
                        'return':
                        '%.3f' % display_reward,
                        'score':
                        '%.3f' % display_score,
                        'obstacle':
                        '%.3f' % display_obstacle,
                        'dist':
                        '%.3f' % display_dist
                    })
                    sys.stdout = sys.__stdout__
                    pbar.update(1)
                    sys.stdout = log_file
        

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters as follow: ")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument('--log', action='store_true', help="Enable logging")
    args = parser.parse_args()

    with open(args.config_path, 'r') as j:
        config = json.load(j)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    algorithm = mySACAlgorithm(
        nums_episodes=config["num_episodes"],
        state_dim=config["state_dim"],
        actor_hidden_dim=config["actor_hidden_dim"],
        critic_hidden_dim=config["critic_hidden_dim"],
        action_dim=config["action_dim"],
        num_actor_hidden_layers=config["num_actor_hidden_layers"],
        num_critic_hidden_layers=config["num_critic_hidden_layers"],
        actor_lr=config["actor_lr"],
        critic_lr=config["critic_lr"],
        alpha_lr=config["alpha_lr"],
        target_entropy=config["target_entropy"],
        tau=config["tau"],
        gamma=config["gamma"],
        device=device
    )
    
    train_offline_policy_agent(algorithm=algorithm,
        num_episodes=config["num_episodes"],
        replay_buffer_size=config["buffer_size"],
        minimal_size=config["minimal_size"],
        batch_size=config["batch_size"],
        config=config["config_name"],
        is_log=args.log)