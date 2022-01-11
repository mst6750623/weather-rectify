import torch
import torch.nn as nn
from .base_classifier import BaseClassifier


class OrdinalRegressionNetwork(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 num_of_classes):
        super(OrdinalRegressionNetwork, self).__init__()
        self.num_of_classes = num_of_classes
        self.classifiers = nn.ModuleList()

        for i in range(self.num_of_classes):
            self.classifiers.append(
                BaseClassifier(in_channels, mid_channels, out_channels))

    def forward(self, x):
        ###每一类的概率####
        probabilities = [
            self.classifiers[i](x) for i in range(self.num_of_classes)
        ]
        return torch.cat(probabilities, dim=1)
