"""
Many places in this file are hard-coded for crazyflies environments.

"""
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
# from environments.quadrotor import QuadrotorEnv

root_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--dir", default='transfer', type=str)
    parser.add_argument("--alg", default="spederv3")  # Alg name (sac, feature_sac)
    parser.add_argument("--agent_path",
                        default=root_dir + "/log/hover-aviary-v0/spederv3/sac_raw_force_input/1")  # Environment name
    parser.add_argument("--log_path",
                        default=root_dir + "/deploy/sample_log")  # Environment name
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=0, type=float)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=2e4, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e3, type=float)  # Max time steps to run environment
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


    # dir_name =
    log_path = f'log/transfer/{args.alg}/{args.dir}/log'
    summary_writer = SummaryWriter(log_path)

    # Store training parameters
    kwargs = vars(args)
    with open(os.path.join(log_path, 'train_params.pkl'), 'wb') as fp:
        pkl.dump(kwargs, fp)

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kwargs = {
        "state_dim": 28,
        "action_dim": 4,
        "action_space": gym.spaces.box.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
        "discount": args.discount,
        "tau": args.tau,
        "hidden_dim": args.hidden_dim,
    }
    kwargs.update(vars(args))

    agent = feature_sac_agent.TransferAgent(**kwargs)
    # agent.actor.load_state_dict(torch.load(os.path.join(args.agent_path, 'best_actor.pth')))
    # agent.critic.load_state_dict(torch.load(os.path.join(args.agent_path, 'best_critic.pth')))
    # agent.feature_mu.load_state_dict(torch.load(os.path.join(args.agent_path, 'best_feature_mu.pth')))

    replay_buffer = buffer.RealDataBuffer(device = args.device)
    replay_buffer.load_all_data(args.log_path)
    #
    timer = util.Timer()

    # best_eval_ret = -1e6

    for t in range(int(args.max_timesteps)):

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            info = agent.train(replay_buffer, batch_size=args.batch_size)

            if (t+1) % 1000 == 0: # add more frequent logging for train stats.
                for key, value in info.items():
                    summary_writer.add_scalar(f'info/{key}', value, t + 1)
                summary_writer.flush()

            steps_per_sec = timer.steps_per_sec(t + 1)
            
            if t >= int(args.max_timesteps) - 5:
                terminal_actor = agent.actor.state_dict()
                terminal_critic = agent.critic.state_dict()
                torch.save(agent.actor.state_dict(), os.path.join(log_path, 'terminal_actor_{}.pth'.format(t)))
                torch.save(agent.critic.state_dict(), os.path.join(log_path, 'terminal_critic_{}.pth'.format(t)))

                if args.alg != 'sac':
                    best_feature_mu = agent.feature_mu.state_dict()
                    torch.save(best_feature_mu, os.path.join(log_path, 'terminal_mu_{}.pth'.format(t)))

            print('Step {}. Steps per sec: {:.4g}.'.format(t + 1, steps_per_sec))

    summary_writer.close()

    print('Total time cost {:.4g}s.'.format(timer.time_cost()))

    torch.save(agent.actor.state_dict(), os.path.join(log_path, 'last_actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(log_path, 'last_critic.pth'))
    if args.alg != 'sac':
        torch.save(agent.feature_mu.state_dict(), os.path.join(log_path, 'last_feature_mu.pth'))
