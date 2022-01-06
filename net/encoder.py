import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(encoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels))

    def forward(self, input):
        return self.layer1(input)


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