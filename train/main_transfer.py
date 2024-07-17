import numpy as np
import torch
import gym
import argparse
import os

from tensorboardX import SummaryWriter

from utils import util, buffer
from agent.sac import sac_agent
from agent.feature_sac import feature_sac_agent
from networks.features import MLPFeatureMu, MLPFeaturePhi

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=0, type=int)
    parser.add_argument("--alg", default="feature_sac")  # Alg name (sac, feature_sac)
    parser.add_argument("--env", default="Pendulum-v1")  # Environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=float)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=5e5, type=float)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--hidden_dim", default=256, type=int)  # Network hidden dims
    parser.add_argument("--feature_dim", default=256, type=int)  # Latent feature dim
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--learn_bonus", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--extra_feature_steps", default=3, type=int)
    args = parser.parse_args()

    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    env.seed(args.seed)
    eval_env.seed(args.seed)
    max_length = env._max_episode_steps

    # setup log
    log_path = f'log/{args.env}/{args.alg}/{args.dir}/{args.seed}'
    summary_writer = SummaryWriter(log_path)

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
    elif args.alg == 'feature_sac':
        kwargs['extra_feature_steps'] = args.extra_feature_steps
        kwargs['feature_dim'] = args.feature_dim
        agent = feature_sac_agent.MLEFeatureAgent(**kwargs)

    log_path = '/home/mht/PycharmProjects/lvrep-rl/log/Pendulum-v1/speder/reproduce_speder/0'
    feature_phi = MLPFeaturePhi(state_dim=state_dim,
                                action_dim=action_dim,
                                feature_dim=args.feature_dim)
    feature_mu = MLPFeatureMu(state_dim=state_dim,
                              feature_dim=args.feature_dim)
    feature_phi.load_state_dict(torch.load(os.path.join(log_path, 'best_feature_phi.pth'), map_location={'cuda:1': 'cuda:0'}))
    feature_mu.load_state_dict(torch.load(os.path.join(log_path, 'best_feature_mu.pth'), map_location={'cuda:1': 'cuda:0'}))

    batch_size = 256
    states = []
    actions = []
    for _ in range(batch_size):
        state = env.reset()
        states.append(state)
        actions.append(env.action_space.sample())
    batch_states = torch.tensor(states)
    batch_actions = torch.tensor(actions)
    phi = feature_phi(batch_states, batch_actions)
    mu = feature_mu(batch_states)
    print(mu, phi)


