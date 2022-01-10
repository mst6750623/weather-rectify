import torch
import torch.nn as nn
import torch.nn.functional as F


class confidenceNetwork(nn.Module):
    def __init__(self, ):
        super(confidenceNetwork, self).__init__()

        self.conv = nn.Sequential(BasicConv2d(58, 64, 3, bn=True, padding=1),
                                  BasicConv2d(64, 64, 3, bn=True, padding=1),
                                  BasicConv2d(64, 128, 3, bn=True, padding=1),
                                  BasicConv2d(128, 128, 3, bn=True, padding=1))
        ##TODO 感觉这里input_channel是不是没改啊 ##
        ###REPO 确实没改###
        self.downsample = nn.Sequential(
            BasicConv2d(58, 128, 1, bn=True, padding=0))

        # to be continue

        ########zx 0108#########
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(128 * 1 * 1, 32),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(32, 2)  ##我感觉是2分类啊##
        )

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv(x)
        x = x + residual
        x = self.pool(x)
        x = self.fc(x)
        return x
        ###########zx 0108############


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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc = nn.Sequential(Flatten(), nn.Dropout(0.2),
                                nn.Linear(37 * 17 * 17, 1024), nn.ReLU(True),
                                nn.Dropout(0.2), nn.Linear(1024, 128),
                                nn.ReLU(True), nn.Dropout(0.2),
                                nn.Linear(128, 1))

    def forward(self, x):
        x = self.fc(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)