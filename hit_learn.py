import gym
import arm_env
import numpy as np
import argparse
import pybullet as p

import torch

from TD3_new import TD3, ReplayBuffer, device, writer

def target_policy(env, obs):
    target_theta = p.calculateInverseKinematics(env.robot_id, env.end_eff_idx, env.targetPos[int(obs[-1])-1])
    delta_theta = target_theta - obs[0:6]
    target_velocity = []
    for i in range(len(delta_theta)):
        if delta_theta[i] > 0.02:
            target_velocity.append(1)
        elif delta_theta[i] < 0.02:
            target_velocity.append(-1)
        else:
            target_velocity.append(0)
    # delta_theta = delta_theta / np.math.pi
    distance_reward = np.exp(-sum([i**2 for i  in delta_theta])/len(delta_theta))
    return target_velocity, distance_reward

def mimic(target_action, action):
    dist = 0
    for i in range(len(action)):
        dist += np.power(target_action[i] - action[i], 2)
    return np.exp(-2*dist/len(action))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_episode_num", default=1e4, type=int)     # How many time steps purely random policy is run for
    parser.add_argument("--max_episode_num", default=1e9, type=float)     # Max time steps to run environment for
    parser.add_argument("--discount", default=0.99, type=float)         # Discount factor
    parser.add_argument("--expl_noise", default=0.1, type=float)		# Std of Gaussian exploration noise
    parser.add_argument("--max_expl_noise", default=1.0, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--min_expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)			# Batch size for both actor and critic
    parser.add_argument("--tau", default=0.005, type=float)             # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)      # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)        # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates
    args = parser.parse_args()

    env = gym.make("ArmHitEnv-v1")
    file_name = "TD3_%s_%s" % ("ArmHitEnv", str(args.seed))

    # Set seeds
    torch.manual_seed(1)
    np.random.seed(1)

    # model config
    state_dim = 6+6+1 # pos, vel, target
    action_dim = 6
    max_action = np.ones(action_dim)
    min_action = -np.ones(action_dim)
    ACTION_BOUND = [min_action, max_action]
    VAR_MIN = args.min_expl_noise
    var = args.max_expl_noise

    #Initial policy
    policy = TD3(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()

    total_timesteps = 0
    episode_num = 0
    done = False
    total_reward = 0
    episode_reward = 0
    episode_timesteps = 0
    success = []
    obs = env.reset()

    while episode_num < args.max_episode_num:
        if done:
            if episode_reward > 10:
                success.append(1)
            else: success.append(0)
            if episode_num % 20 == 0 and episode_num != 0:
                print('Successful Rate: ', sum(success[-100:]), '%')
                writer.add_scalar('success_rate', sum(success[-100:]), episode_num)
            if episode_num != 0:
                writer.add_scalar('episode_reward', episode_reward, episode_num)
            
            # Reset environment
            obs = env.reset()

            if total_timesteps != 0:
                policy.train(replay_buffer, episode_timesteps, total_timesteps, args.batch_size, args.discount, args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
            
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            if episode_num % 50 == 0:
                policy.save("%s" % (file_name), directory="./pytorch_models")
                print('Model saved !')
        
        action = policy.select_action(obs)
        target_action, distance_reward = target_policy(env, obs)
        action = np.clip(np.random.normal(action, var), *ACTION_BOUND)
        if episode_num>args.start_episode_num:
            var = max([var*.9999, VAR_MIN])
        
        new_obs, reward, done, _ = env.step(action)
        reward = reward + mimic(target_action, action) + distance_reward
        episode_reward += reward
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        print("reward:", reward)
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
    
    policy.save("%s" % (file_name), directory="./pytorch_models")