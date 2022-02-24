import torch
import torch.nn as nn


class TimeReflect(nn.Module):
    def __init__(self, in_channels=8, out_channels=4):
        super(TimeReflect, self).__init__()
        self.net = nn.Sequential(nn.Linear(in_channels, out_channels))

    def forward(self, x):
        return self.net(x)