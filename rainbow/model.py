import torch
from torch import nn


class Agent(nn.Module):
    def __init__(self, *args):
        super(Agent, self).__init__()
        