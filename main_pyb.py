import numpy as np
import torch
import gym
import gymnasium
import argparse
import os
import datetime
import yaml
import pickle as pkl

from tensorboardX import SummaryWriter
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary

from train.utils import util, buffer
from train.agent.sac import sac_agent
from train.agent.feature_sac import feature_sac_agent

root_dir = os.path.dirname(os.path.abspath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Gymnasium2GymWrapper(gymnasium.core.Wrapper):
    """Wrapper to translate the Gymnasium interface into Gym interface."""

    def __init__(self, env):
        super().__init__(env)

        # Translate action space
        if isinstance(self.env.action_space, gymnasium.spaces.Discrete):
            self.env.action_space = gym.spaces.Discrete(self.env.action_space.n)
        elif isinstance(self.env.action_space, gymnasium.spaces.Box):
            self.env.action_space = gym.spaces.Box(
                high=self.env.action_space.high,
                low=self.env.action_space.low,
                shape=self.env.action_space.shape,
                dtype=self.env.action_space.dtype)

        # Translate observation space
        if isinstance(self.env.observation_space, gymnasium.spaces.Discrete):
            self.env.observation_space = gym.spaces.Discrete(self.env.observation_space.n)
        elif isinstance(self.env.observation_space, gymnasium.spaces.Box):
            self.env.observation_space = gym.spaces.Box(
                high=self.env.observation_space.high,
                low=self.env.observation_space.low,
                shape=self.env.observation_space.shape,
                dtype=self.env.observation_space.dtype)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, max(terminated, truncated), info

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument("--dir", default='sac_raw_force_input', type=str)
    parser.add_argument("--alg", default="spederv3")  # Alg name (sac, feature_sac)
    parser.add_argument("--env", default="hover-aviary-v0")  # Environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=0, type=float)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=2e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=16e5, type=float)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=256, type=int)  # Network hidden dims
    parser.add_argument("--feature_dim", default=1024, type=int)  # Latent feature dim
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--learn_bonus", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--extra_feature_steps", default=3, type=int)
    args = parser.parse_args()

    # load env params

    env = gymnasium.make(args.env)
    env = gymnasium.wrappers.transform_reward.TransformReward(env, lambda r: 0.2 * r)
    eval_env = gymnasium.make(args.env)
    env = Gymnasium2GymWrapper(env)
    eval_env = Gymnasium2GymWrapper(eval_env)
    
   
    # env.seed(args.seed)
    # eval_env.seed(args.seed)
    # max_length = env._max_episode_steps

    # setup log
    # dir_name =
    log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.seed}/log'
    summary_writer = SummaryWriter(log_path)

    # Store training parameters
    kwargs = vars(args)
    with open(os.path.join(log_path, 'train_params.pkl'), 'wb') as fp:
        pkl.dump(kwargs, fp)

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_space": env.action_space,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_dim": args.hidden_dim,
    }

    # Initialize policy
    if args.alg == "sac":
        agent = sac_agent.SACAgent(**kwargs)
    elif args.alg == 'mle':
        kwargs['extra_feature_steps'] = args.extra_feature_steps
        kwargs['feature_dim'] = args.feature_dim
        agent = feature_sac_agent.MLEFeatureAgent(**kwargs)
    elif args.alg == 'speder':
        kwargs['extra_feature_steps'] = args.extra_feature_steps
        kwargs['feature_dim'] = args.feature_dim
        agent = feature_sac_agent.SPEDERAgent(**kwargs)
    elif args.alg == 'spederv2':
        kwargs['extra_feature_steps'] = args.extra_feature_steps
        kwargs['feature_dim'] = args.feature_dim
        agent = feature_sac_agent.SPEDERAgentV2(**kwargs)
    elif args.alg == 'spederv3':
        kwargs['extra_feature_steps'] = args.extra_feature_steps
        kwargs['feature_dim'] = args.feature_dim
        agent = feature_sac_agent.SPEDERAgentV3Mel(**kwargs)

    replay_buffer = buffer.ReplayBuffer(state_dim, action_dim, device=args.device)

    # Evaluate untrained policy
    evaluations = [util.eval_policy(agent, eval_env)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    timer = util.Timer()

    best_eval_ret = -1e6

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, explore=True)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) # if episode_timesteps < max_length else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            info = agent.train(replay_buffer, batch_size=args.batch_size)

            if (t+1) % 1000 == 0: # add more frequent logging for train stats.
                for key, value in info.items():
                    summary_writer.add_scalar(f'info/{key}', value, t + 1)
                summary_writer.flush()

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            steps_per_sec = timer.steps_per_sec(t + 1)
            eva_ret = util.eval_policy(agent, eval_env)
            evaluations.append(eva_ret)

            if t >= args.start_timesteps:
                info['evaluation'] = eva_ret
                for key, value in info.items():
                    summary_writer.add_scalar(f'info/{key}', value, t + 1)
                summary_writer.flush()

            if eva_ret > best_eval_ret:
                best_actor = agent.actor.state_dict()
                best_critic = agent.critic.state_dict()
                torch.save(best_actor, os.path.join(log_path, 'best_actor.pth'))
                torch.save(best_critic, os.path.join(log_path, 'best_critic.pth'))

                if args.alg != 'sac':
                    if not args.alg.endswith('v3'):
                        best_feature_phi = agent.feature_phi.state_dict()
                        torch.save(best_feature_phi, os.path.join(log_path, 'best_feature_phi.pth'))
                    best_feature_mu = agent.feature_mu.state_dict()
                    torch.save(best_feature_mu, os.path.join(log_path, 'best_feature_mu.pth'))
            
            if t >= int(args.max_timesteps) - 5:
                terminal_actor = agent.actor.state_dict()
                terminal_critic = agent.critic.state_dict()
                torch.save(best_actor, os.path.join(log_path, 'terminal_actor_{}.pth'.format(t)))
                torch.save(best_critic, os.path.join(log_path, 'terminal_critic_{}.pth'.format(t)))

                if args.alg != 'sac':
                    if not args.alg.endswith('v3'):
                        best_feature_phi = agent.feature_phi.state_dict()
                        torch.save(best_feature_phi, os.path.join(log_path, 'terminal_phi_{}.pth'.format(t)))
                    best_feature_mu = agent.feature_mu.state_dict()
                    torch.save(best_feature_mu, os.path.join(log_path, 'terminal_mu_{}.pth'.format(t)))

            print('Step {}. Steps per sec: {:.4g}.'.format(t + 1, steps_per_sec))

    summary_writer.close()

    print('Total time cost {:.4g}s.'.format(timer.time_cost()))

    torch.save(agent.actor.state_dict(), os.path.join(log_path, 'last_actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(log_path, 'last_critic.pth'))
    if args.alg != 'sac':
        if not args.alg.endswith('v3'):
            torch.save(agent.feature_phi.state_dict(), os.path.join(log_path, 'last_feature_phi.pth'))
        torch.save(agent.feature_mu.state_dict(), os.path.join(log_path, 'last_feature_mu.pth'))
