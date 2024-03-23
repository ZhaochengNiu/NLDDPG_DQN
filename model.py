import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hyperparameter as hp


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        # 输出用的 tanh
        # tanh 函数的输出范围为 [-1, 1]，这与许多动作空间的要求相符合。例如，如果动作是连续的，并且需要落在某个范围内（如 [-1, 1] 或 [0, 1]），那么使用 tanh 可以将输出映射到合适的范围内。
        a = F.tanh(self.l3(s))
        return a

class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, hidden_width)
        self.l3 = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
