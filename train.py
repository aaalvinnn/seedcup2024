from env import Env
from my_algorithm import myPPOAlgorithm
from my_algorithm_SAC import mySACAlgorithm
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
import csv
import time

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

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
    print(f"Step: {env.step_num}, dest_dist: {env.get_dis()}, is_Obstacle: {obstacle_contact}, Reward: {reward}\n\
        State: {env.get_observation()[0]}\nAction: {action}\n")


def train_offline_policy_agent(algorithm: mySACAlgorithm, num_episodes, replay_buffer_size, minimal_size, batch_size, config, seed, is_log):
    replay_buffer = ReplayBuffer(replay_buffer_size)
    output_dir = None

    if is_log:
        cur_time = datetime.now()
        output_fir_name = os.path.join("output", cur_time.strftime("%m%d"), cur_time.strftime("%H%M") + "_" + config)
        output_dir = os.path.join(os.path.dirname(__file__), output_fir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        writer = SummaryWriter(os.path.join(output_dir, 'SummaryWriter'))
        log_file = open(os.path.join(output_dir, 'log.txt'), 'w+')
        sys.stdout = log_file

    env = Env(is_senior=True,seed=seed,gui=False)
    done = False
    total_score_list = []
    score100_best = 0
    total_reward_list = []
    total_obstacle_list = []
    total_dist_list = []
    epoch_actor_best = 0

    for i in range(int(num_episodes/100)):
        with tqdm(total=100, desc='Iteration %d' % i) as pbar:
            for i_episode in range(100):
                algorithm.reset(env.reset()[0], seed)
                epoch = 100 * i + i_episode + 1
                score = 0
                total_reward = 0
                total_obstacle = 0
                actor_loss, alpha_loss, actor_lr, critic_lr, _ = 0, 0, 0, 0, 0
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
                    reward = algorithm.reward_total_13_test(env.get_dis(), pre_dist, env.is_obstacle_contact(), total_obstacle, env.get_step_now(), env.get_score())
                    replay_buffer.add(state, action, reward, new_state, done)
                    total_reward += reward
                    score += env.success_reward
                    pre_dist = env.get_dis()
                    state = new_state
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        actor_loss, alpha_loss, actor_lr, critic_lr, _ = algorithm.update(transition_dict)

                    if is_log:
                        env_step_log(env, action, reward, env.is_obstacle_contact())
                        writer.add_scalar('Actor Loss', actor_loss, (epoch-1)*100+env.get_step_now())
                        writer.add_scalar('Alpha Loss', alpha_loss, (epoch-1)*100+env.get_step_now())
                        writer.add_scalar('Actor LR', actor_lr, (epoch-1)*100+env.get_step_now())
                        writer.add_scalar('Critic LR', critic_lr, (epoch-1)*100+env.get_step_now())

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
                    # import wandb
                    # wandb.log({
                    #     'Total Reward': total_reward,
                    #     'Score': score,
                    #     'End Distance': env.get_dis(),
                    #     'Obstacle Contact Num': total_obstacle
                    # }, step=epoch)

                    # model saving
                    # torch.save(algorithm.actor.state_dict(), os.path.join(os.path.dirname(__file__), 'model.pth'))  # 放个在工程根目录下方便test.py测试
                    torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor.pth'))
                    torch.save(algorithm.critic_1.state_dict(), os.path.join(output_dir, 'critic_1.pth'))
                    torch.save(algorithm.critic_2.state_dict(), os.path.join(output_dir, 'critic_2.pth'))

                    if (epoch>=100):
                        score100_best = np.mean(total_score_list[-100:]) if np.mean(total_score_list[-100:]) > score100_best else score100_best
                        if (score100_best <= np.mean(total_score_list[-100:])):
                            epoch_actor_best = epoch
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
                    if (epoch >= 100):
                        display_reward = np.mean(total_reward_list[-100:])
                        display_score = np.mean(total_score_list[-100:])
                        display_obstacle = np.mean(total_obstacle_list[-100:])
                        display_dist = np.mean(total_dist_list[-100:])
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
    # writer.close()
    # if is_log:
    #     wandb.save(os.path.join(output_dir, 'actor_best.pth'))
    # # 返回最佳test得分，平均总得分，平均奖励回报，平均碰到障碍次数，平均结束距离
    return score100_best, sum(total_score_list)/len(total_score_list), sum(total_reward_list)/len(total_reward_list), sum(total_obstacle_list)/len(total_obstacle_list), sum(total_dist_list)/len(total_dist_list)

def train_online_policy_agent(algorithm: myPPOAlgorithm, num_episodes, config, seed, is_log):
    output_dir = None
    if is_log:
        cur_time = datetime.now()
        output_fir_name = os.path.join("output", cur_time.strftime("%m%d"), cur_time.strftime("%H%M") + "_" + config)
        output_dir = os.path.join(os.path.dirname(__file__), output_fir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        writer = SummaryWriter(os.path.join(output_dir, 'SummaryWriter'))
        log_file = open(os.path.join(output_dir, 'log.txt'), 'w+')
        sys.stdout = log_file

    env = Env(is_senior=True,seed=seed,gui=False)
    done = False
    total_score_list = []
    score100_best = 0
    total_reward_list = []
    total_obstacle_list = []
    total_dist_list = []
    epoch_actor_best = 0

    for i in range(int(num_episodes/100)):
        with tqdm(total=100, desc='Iteration %d' % i) as pbar:
            for i_episode in range(100):
                algorithm.reset(env.reset()[0])
                epoch = 100 * i + i_episode + 1
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
                    reward = algorithm.reward_total_13_test(env.get_dis(), pre_dist, env.is_obstacle_contact(), total_obstacle, env.get_step_now(), env.get_score())
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(new_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    total_reward += reward
                    score += env.success_reward
                    pre_dist = env.get_dis()
                    state = new_state
                    if is_log:
                        env_step_log(env, action, reward, env.is_obstacle_contact())

                    if (env.is_obstacle_contact()):
                        total_obstacle += 1
                        # break

                total_reward_list.append(total_reward)
                total_score_list.append(score)
                total_obstacle_list.append(total_obstacle)
                total_dist_list.append(env.get_dis())
                actor_loss = 0
                critic_loss = 0
                actor_loss, critic_loss, actor_lr, critic_lr = algorithm.update(transition_dict)
                print(f"Train_{epoch} completed. steps:", env.step_num, "Distance:", env.get_dis(), "Score:", score, "Reward:", total_reward, "n_Obstacle: ", total_obstacle, "Actor Loss:", actor_loss, "Critic Loss:", critic_loss)

                # Tensorboard logging
                if is_log:
                    writer.add_scalar('Actor Loss', actor_loss, epoch)
                    writer.add_scalar('Critic Loss', critic_loss, epoch)
                    writer.add_scalar('Total Reward', total_reward, epoch)
                    writer.add_scalar('Score', score, epoch)
                    writer.add_scalar('End Distance', env.get_dis(), epoch)
                    writer.add_scalar('Obstacle Contact Num', total_obstacle, epoch)
                    writer.add_scalar('Actor LR', actor_lr, epoch)
                    writer.add_scalar('Critic LR', critic_lr, epoch)
                    # import wandb
                    # wandb.log({
                    #     'Actor Loss': actor_loss,
                    #     'Critic Loss': critic_loss,
                    #     'Total Reward': total_reward,
                    #     'Score': score,
                    #     'End Distance': env.get_dis(),
                    #     'Obstacle Contact Num': total_obstacle
                    # }, step=epoch)

                    # model saving
                    # torch.save(algorithm.actor.state_dict(), os.path.join(os.path.dirname(__file__), 'model.pth'))  # 放个在工程根目录下方便test.py测试
                    torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor.pth'))
                    torch.save(algorithm.critic.state_dict(), os.path.join(output_dir, 'critic.pth'))

                    if (epoch>=100):
                        score100_best = np.mean(total_score_list[-100:]) if np.mean(total_score_list[-100:]) > score100_best else score100_best
                        if (score100_best <= np.mean(total_score_list[-100:])):
                            epoch_actor_best = epoch
                            torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor_best.pth'))
                            torch.save(algorithm.critic.state_dict(), os.path.join(output_dir, 'critic_best.pth'))
                            # torch.save(algorithm.actor.state_dict(), os.path.join(os.path.dirname(__file__), 'model_best.pth'))  # 放个在工程根目录下方便test.py测试
                    if (epoch == 1):
                        torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor_init.pth'))    # 由于随机性，保存一下初始权重，提供预训练模型
                        torch.save(algorithm.critic.state_dict(), os.path.join(output_dir, 'critic_init.pth'))

                    display_reward = total_reward_list[-1]
                    display_score = total_score_list[-1]
                    display_obstacle = total_obstacle_list[-1]
                    display_dist = env.get_dis()
                    if (epoch >= 100):
                        display_reward = np.mean(total_reward_list[-100:])
                        display_score = np.mean(total_score_list[-100:])
                        display_obstacle = np.mean(total_obstacle_list[-100:])
                        display_dist = np.mean(total_dist_list[-100:])
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
    writer.close()
    # if is_log:
    #     wandb.save(os.path.join(output_dir, 'actor_best.pth'))
    # 返回最佳test得分，平均总得分，平均奖励回报，平均碰到障碍次数，平均结束距离
    return score100_best, sum(total_score_list)/len(total_score_list), sum(total_reward_list)/len(total_reward_list), sum(total_obstacle_list)/len(total_obstacle_list), sum(total_dist_list)/len(total_dist_list)

def seed_everything(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_config(config_file_path, is_log, seed):
    # TRY NOT TO MODIFY: seeding
    seed_everything(seed)

    with open(config_file_path, 'r') as j:
        config = json.load(j)
    
    # if is_log:
    #     import wandb
    #     wandb.init(
    #         project="robot_SAC",  # 设置你的项目名称
    #         config=config,
    #         name=f"{config['config_name']}__{seed}__{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    #         save_code=True
    #     )
    
    config_name = config["config_name"]
    score100 = 0
    mean_score = 0
    mean_reward = 0
    mean_obstacle = 0
    mean_end_dist = 0
    sys.stdout = sys.__stdout__
    print(f"***************{datetime.now()} config: {config_name}***************\n")

    if (config["algorithm"] == "PPO"):
        algorithm = myPPOAlgorithm(
            nums_episodes=config["num_episodes"],
            state_dim=config["state_dim"],
            actor_hidden_dim=config["actor_hidden_dim"],
            critic_hidden_dim=config["critic_hidden_dim"],
            action_dim=config["action_dim"],
            actor_lr=config["actor_lr"],
            critic_lr=config["critic_lr"],
            lmbda=config["lmbda"],
            epochs=config["epochs"],
            eps=config["eps"],
            gamma=config["gamma"],
            residual_strength=config.get("residual_strength"),
            device=device,
            dropout=config.get("dropout"),
            n_state_steps=config.get("n_state_steps"),
            num_actor_hidden_layers=config["num_actor_hidden_layers"],
            num_critic_hidden_layers=config["num_critic_hidden_layers"],
            actor_pretrained_model=config.get("actor_pretrained_model"),
            critic_pretrained_model=config.get("critic_pretrained_model"),
            isTrain=True,
            a=config.get("a"),
            b=config.get("b"),
            c=config.get("c"),
            d=config.get("d"),
            e=config.get("e")
        )
        score100, mean_score, mean_reward, mean_obstacle, mean_end_dist = train_online_policy_agent(algorithm, config["num_episodes"], config["config_name"], seed, is_log)
        
    elif (config["algorithm"] == "SAC"):
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
            device=device,
            pretrained_actor=config.get("pretrained_actor"),
            pretrained_critic_1=config.get("pretrained_critic_1"),
            pretrained_critic_2=config.get("pretrained_critic_2"),
            a=config.get("a"),
            b=config.get("b"),
            c=config.get("c"),
            d=config.get("d"),
            e=config.get("e")
        )
        score100, mean_score, mean_reward, mean_obstacle, mean_end_dist = train_offline_policy_agent(algorithm=algorithm,
            num_episodes=config["num_episodes"],
            replay_buffer_size=config["buffer_size"],
            minimal_size=config["minimal_size"],
            batch_size=config["batch_size"],
            config=config["config_name"],
            seed=seed,
            is_log=is_log)
        
    # if is_log:
    #     wandb.finish()
    return config_name, score100, mean_score, mean_reward, mean_obstacle, mean_end_dist, datetime.now()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters as follow: ")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the JSON configuration file")
    parser.add_argument('--log', action='store_true', help="Enable logging")
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()

    with open(args.output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Config File', 'Score 100', 'Mean Score', 'Mean Reward', 'Mean Obstacle', 'Mean End Distance', 'Time'])

        if os.path.isfile(args.config_path):
            # 处理单个配置文件
            config_name, score100, mean_score, mean_reward, mean_obstacle, mean_end_dist, time = train_one_config(args.config_path, args.log, args.seed)
            writer.writerow([config_name, score100, mean_score, mean_reward, mean_obstacle, mean_end_dist, time])
        
        elif os.path.isdir(args.config_path):
            # 处理目录中的多个配置文件
            config_files = [os.path.join(args.config_path, f) for f in os.listdir(args.config_path) if f.endswith('.json')]
            for config_file in config_files:
                config_name, score100, mean_score, mean_reward, mean_obstacle, mean_end_dist, time = train_one_config(config_file, args.log, args.seed)
                writer.writerow([config_name, score100, mean_score, mean_reward, mean_obstacle, mean_end_dist, time])

    print(f"Results have been written to {args.output_csv}")
    