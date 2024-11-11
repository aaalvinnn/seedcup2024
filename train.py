from env import Env
from my_algorithm import myPPOAlgorithm, myREINFORCEAlgorithm
from team_algorithm import PPOAlgorithm, MyCustomAlgorithm
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import sys
from tqdm import tqdm

def env_step_log(env, action, reward):
    print(f"Step: {env.step_num}, Distance: {env.get_dis()}, Reward: {reward}, State: {env.get_observation()[0][:6]}, Action: {action}")

def main(algorithm, num_episodes):
    cur_time = datetime.now()
    output_fir_name = os.path.join("output", cur_time.strftime("%Y%m%d%H%M"))
    output_dir = os.path.join(os.path.dirname(__file__), output_fir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    writer = SummaryWriter(os.path.join(output_dir, 'SummaryWriter'))
    log_file = open(os.path.join(output_dir, 'log.txt'), 'w+')
    sys.stdout = log_file

    env = Env(is_senior=True,seed=100,gui=False)
    done = False
    best_return = 0

    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                epoch = num_episodes / 10 * i + i_episode + 1
                score = 0
                total_reward = 0
                done = False
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                print(f"******** Episode:", epoch, "*******")

                while not done:
                    action = algorithm.get_action(state[0])
                    _ = env.step(action)
                    new_state = env.get_observation()
                    done = env.terminated
                    # reward = 1 / (1 + 4*env.get_dis())
                    # reward = 1 / (env.get_dis() + 0.01)**2
                    reward = 1 - (env.get_dis() - 0.2)/0.15
                    env_step_log(env, action, reward)
                    total_reward += reward
                    score += env.success_reward
                    transition_dict['states'].append(state[0])
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(new_state[0])
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                actor_loss = 0
                critic_loss = 0
                actor_loss, critic_loss = algorithm.update(transition_dict)
                print(f"Train_{i} completed. steps:", env.step_num, "Distance:", env.get_dis(), "Score:", score, "Reward:", total_reward, "Actor Loss:", actor_loss, "Critic Loss:", critic_loss)

                # Tensorboard logging
                writer.add_scalar('Actor Loss', actor_loss, epoch)
                writer.add_scalar('Critic Loss', critic_loss, epoch)
                writer.add_scalar('Total Reward', total_reward, epoch)
                writer.add_scalar('Score', score, epoch)

                # model saving
                torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor.pth'))
                torch.save(algorithm.critic.state_dict(), os.path.join(output_dir, 'critic.pth'))
                if (total_reward > best_return):
                    best_return = total_reward
                    torch.save(algorithm.actor.state_dict(), os.path.join(output_dir, 'actor_best.pth'))
                    torch.save(algorithm.critic.state_dict(), os.path.join(output_dir, 'critic_best.pth'))

                pbar.set_postfix({
                    'episode':
                    '%d' % (epoch),
                    'return':
                    '%.3f' % total_reward
                })
                sys.stdout = sys.__stdout__
                pbar.update(1)
                sys.stdout = log_file
        

    env.close()

if __name__ == "__main__":
    actor_lr = 1e-3
    critic_lr = 5e-3
    num_episodes = 1000
    state_dim = 12
    hidden_dim = 128
    action_dim = 6
    gamma = 0.1
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    algorithm = myPPOAlgorithm(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)
    # algorithm = myREINFORCEAlgorithm(state_dim, hidden_dim, action_dim, actor_lr, gamma, device)
    # algorithm = PPOAlgorithm()
    main(algorithm, num_episodes)