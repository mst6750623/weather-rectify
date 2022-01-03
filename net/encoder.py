import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(encoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels))

    def forward(self, input):
        return self.layer1(input)
