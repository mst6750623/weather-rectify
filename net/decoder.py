import torch
import torch.nn as nn
from confidence import BasicConv2d


class decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels = 2):
        super(decoder, self).__init__()

        self.layer1 = BasicConv2d(in_channels, mid_channels, 3, bn = True, padding = 1)
        self.upsample = nn.Upsample(size = (69, 73), mode = 'bilinear')
        self.layer2 = nn.Sequential(
            BasicConv2d(mid_channels, mid_channels, 3, bn = True, padding = 1),
            BasicConv2d(mid_channels, out_channels, 3, bn = True, padding = 1),
        )



    def forward(self, x):
        x = self.layer1(x)
        x = self.upsample(x)
        x = self.layer2(x)
        return x
