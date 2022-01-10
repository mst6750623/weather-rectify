import torch
import torch.nn as nn
from confidence import BasicConv2d
#########zx 0110##########
class encoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 mean = 0, std = 1e-1):
        super(encoder, self).__init__()

        ###args for Gaussian####
        self.mean = mean
        self.std = std

        #self.layer1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels))
        self.layer1 = nn.Sequential(
            BasicConv2d(in_channels, mid_channels, 1, bn = True),
            BasicConv2d(mid_channels, mid_channels, 3, bn = True, padding = 1)
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2),
            BasicConv2d(mid_channels, out_channels, 3, bn = True, padding = 1),
            BasicConv2d(out_channels, out_channels, 3, bn = True, padding = 1)
        )

    def forward(self, x):
        x = self.layer1(x)

        if self.training:
            x = x + torch.normal(
                mean = torch.full(x.size(), fill_value = self.mean),
                std = self.std
            ).to(x.device)

        x = self.layer2(x)
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