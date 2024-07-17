import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as td

import socket
from train import CUDA_DEVICE_WORKSTATION
device_name = socket.gethostname()
if device_name.startswith('naliseas'):
	device = torch.device(CUDA_DEVICE_WORKSTATION if torch.cuda.is_available() else "cpu")
else:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLPFeaturePhi(nn.Module):
    """
    Discrete encoder

    s,a -> phi OR s' -> mu
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            feature_dim=256
    ):
        super(MLPFeaturePhi, self).__init__()

        self.feature_dim = feature_dim

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)  # get logits

    def forward(self, state, action):
        """
        get logits
        """
        x = torch.cat([state, action], axis=-1)
        z = F.elu(self.l1(x))
        z = F.elu(self.l2(z))
        logit = torch.arctan(self.l3(z))
        return logit


class MLPFeatureMu(nn.Module):
    """
    Discrete encoder

    s,a -> phi OR s' -> mu
    """

    def __init__(
            self,
            state_dim,
            hidden_dim=256,
            feature_dim=256
    ):
        super(MLPFeatureMu, self).__init__()

        self.feature_dim = feature_dim

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)  # get logits

    def forward(self, state):
        """
        get logits
        """
        # x = torch.cat([state, action], axis=-1)
        z = F.elu(self.l1(state))
        z = F.elu(self.l2(z))
        logit = 1 / self.feature_dim * torch.arctan(self.l3(z))
        return logit