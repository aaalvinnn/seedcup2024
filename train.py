from env import Env
from my_algorithm import myPPOAlgorithm, myREINFORCEAlgorithm
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

# 定义经验重放队列，PPO不需要经验重放
replay_buffer_size = 1#(*100)
replay_buffer = {'states': collections.deque(maxlen=replay_buffer_size*100),
                 'actions': collections.deque(maxlen=replay_buffer_size*100),
                 'next_states': collections.deque(maxlen=replay_buffer_size*100),
                 'rewards': collections.deque(maxlen=replay_buffer_size*100),
                 'dones': collections.deque(maxlen=replay_buffer_size*100)}

def env_step_log(env, action, reward, obstacle_contact):
    print(f"Step: {env.step_num}, Distance: {env.get_dis()}, is_Obstacle: {obstacle_contact}, Reward: {reward}, State: {env.get_observation()[0][:6]}, Action: {action}")

def main(algorithm, num_episodes, config, is_log):
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
    best_return = 0
    total_reward_list = []

    for i in range(20):
        with tqdm(total=int(num_episodes/20), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/20)):
                epoch = num_episodes / 20 * i + i_episode + 1
                score = 0
                total_reward = 0
                done = False
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                pre_dist = env.get_dis()

                print(f"******** Episode:", epoch, "*******")
                while not done:
                    action = algorithm.get_action(state[0])
                    _ = env.step(action)
                    new_state = env.get_observation()
                    done = env.terminated
                    reward = reward_algorithm.reward_total_8_1(env.get_dis(), pre_dist, env.is_obstacle_contact())
                    if is_log:
                        env_step_log(env, action, reward, env.is_obstacle_contact())
                    total_reward += reward
                    score += env.success_reward
                    pre_dist = env.get_dis()
                    transition_dict['states'].append(state[0])
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(new_state[0])
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                
                for key in replay_buffer.keys():
                    replay_buffer[key].extend(transition_dict[key])

                total_reward_list.append(total_reward)
                actor_loss = 0
                critic_loss = 0
                if len(replay_buffer['states']) >= replay_buffer_size*100:
                    batch = {k: list(replay_buffer[k]) for k in replay_buffer.keys()}
                    actor_loss, critic_loss = algorithm.update(batch)
                print(f"Train_{i} completed. steps:", env.step_num, "Distance:", env.get_dis(), "Score:", score, "Reward:", total_reward, "Actor Loss:", actor_loss, "Critic Loss:", critic_loss)

                # Tensorboard logging
                if is_log:
                    writer.add_scalar('Actor Loss', actor_loss, epoch)
                    writer.add_scalar('Critic Loss', critic_loss, epoch)
                    writer.add_scalar('Total Reward', total_reward, epoch)
                    writer.add_scalar('Score', score, epoch)
                    writer.add_scalar('End Distance', env.get_dis(), epoch)

                    # model saving
                    torch.save(algorithm.actor.state_dict(), os.path.join(os.path.dirname(__file__), 'model.pth'))  # 放个在工程根目录下方便test.py测试
                    torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor.pth'))
                    torch.save(algorithm.critic.state_dict(), os.path.join(output_dir, 'critic.pth'))

                    if (total_reward > best_return):
                        best_return = total_reward
                        torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor_best.pth'))
                        torch.save(algorithm.critic.state_dict(), os.path.join(output_dir, 'critic_best.pth'))
                        torch.save(algorithm.actor.state_dict(), os.path.join(os.path.dirname(__file__), 'model_best.pth'))  # 放个在工程根目录下方便test.py测试
                    if (epoch == 1):
                        torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor_init.pth'))    # 由于随机性，保存一下初始权重，提供预训练模型
                        torch.save(algorithm.critic.state_dict(), os.path.join(output_dir, 'critic_init.pth'))
                    elif (epoch == 1000):
                        torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor_1000.pth'))
                        torch.save(algorithm.critic.state_dict(), os.path.join(output_dir, 'critic_1000.pth'))
                    elif (epoch == 1500):
                        torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor_1500.pth'))
                        torch.save(algorithm.critic.state_dict(), os.path.join(output_dir, 'critic_1500.pth'))

                    display_reward = 0
                    if epoch % (num_episodes/20) == 0:
                        display_reward = np.mean(total_reward_list[-int(num_episodes/20):])
                    else:
                        display_reward = total_reward
                    pbar.set_postfix({
                        'episode':
                        '%d' % (epoch),
                        'return':
                        '%.3f' % display_reward
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

    algorithm = myPPOAlgorithm(
    config["num_episodes"],
    config["state_dim"],
    config["actor_hidden_dim"],
    config["critic_hidden_dim"],
    config["action_dim"],
    config["actor_lr"],
    config["critic_lr"],
    config["lmbda"],
    config["epochs"],
    config["eps"],
    config["gamma"],
    device,
    num_actor_hidden_layers=config["num_actor_hidden_layers"],
    num_critic_hidden_layers=config["num_critic_hidden_layers"]
    # actor_pretrained_model=config["actor_pretrained_model"],
    # critic_pretrained_model=config["critic_pretrained_model"]
)
    
    main(algorithm, config["num_episodes"], config["config_name"], args.log)