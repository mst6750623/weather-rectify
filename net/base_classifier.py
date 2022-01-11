import torch
import torch.nn as nn
from .confidence import (Flatten, BasicConv2d)


#####含residual connection的conv，预测单一probability#####
class BaseClassifier(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(BaseClassifier, self).__init__()

        self.conv = nn.Sequential(
            BasicConv2d(in_channels, mid_channels, 3, padding=1),
            BasicConv2d(mid_channels, mid_channels, 3, padding=1),
            BasicConv2d(mid_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, 1))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        ####residual connection时用这种方式调整#channel######
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels))

        self.ac = nn.ReLU(True)

        self.fc = nn.Sequential(Flatten(), nn.Dropout(0.5),
                                nn.Linear(out_channels, 1), nn.Sigmoid())

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv(x)
        x = self.ac(x + residual)
        x = self.pool(x)
        x = self.fc(x)
        return x