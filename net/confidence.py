import torch
import torch.nn as nn
import torch.nn.functional as F


class confidenceNetwork(nn.Module):
    def __init__(self, ):
        super(confidenceNetwork, self).__init__()
        self.conv = nn.Sequential(BasicConv2d(58, 64, 3, bn=True, padding=1),
                                  BasicConv2d(58, 64, 3, bn=True, padding=1),
                                  BasicConv2d(58, 64, 3, bn=True, padding=1),
                                  BasicConv2d(58, 64, 3, bn=True, padding=1))

        self.downsample = nn.Sequential(
            BasicConv2d(37, 128, 1, bn=True, padding=0))

        # to be continue


"""------- single conv -------"""


class BasicConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 relu=True,
                 bn=False,
                 **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              bias=False,
                              **kwargs)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            #
            x = F.leaky_relu(x, inplace=True)
        return x
