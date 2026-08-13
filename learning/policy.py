"""The behaviour-cloning policy network.

Lives here rather than in the notebook so that play.py and the notebook load
the *same* architecture — ``state_dict`` stores weights only, so both sides
must agree on the shape or loading fails.

Import it in the notebook instead of redefining the class:

    from learning.policy import Policy
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(13, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x
