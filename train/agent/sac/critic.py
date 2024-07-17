"""
We adapt the code from https://github.com/denisyarats/pytorch_sac
"""

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from train.utils import util


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()

        self.Q1 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = util.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(util.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class CriticwithPhi(nn.Module):
    """
    Critic with random fourier features
    """

    def __init__(
            self,
            input_dim,
            feature_dim,
            hidden_dim=256,
            final_layer_grad=True,
    ):
        super().__init__()

        # Q1
        self.l1 = nn.Linear(input_dim, hidden_dim)  # random feature
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)
        self.final_l1 = nn.Linear(feature_dim, 1)
        self.final_l2 = nn.Linear(feature_dim, 1)
        self.apply(util.weight_init)

        if not final_layer_grad:
            self.final_l1.requires_grad(False)
            self.final_l2.requires_grad(False)

    def get_feature(self, state, action):
        x = torch.cat([state, action], axis=-1)
        f = F.elu(self.l1(x))  # F.relu(self.l1(x))
        # q1 = q1.reshape([batch_size, self.num_noise, -1]).mean(dim=1)
        f = F.elu(self.l2(f))  # F.relu(self.l2(q1))
        f = F.tanh(self.l3(f))

        return f

    def forward(self, state, action):
        f = self.get_feature(state, action)
        q1 = self.final_l1(f)
        q2 = self.final_l2(f)
        return q1, q2
