import torch
import torch.nn as nn
from .confidence import BasicConv2d


#########zx 0110##########
class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=32,
                 out_channels=64,
                 mean=0,
                 std=1e-1,
                 training=True):
        super(Encoder, self).__init__()

        ###args for Gaussian####
        self.mean = mean
        self.std = std

        #self.layer1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels))
        self.layer1 = nn.Sequential(
            BasicConv2d(in_channels, mid_channels, 1, bn=True),
            BasicConv2d(mid_channels, mid_channels, 3, bn=True, padding=1))
        self.layer2 = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(32),
            BasicConv2d(mid_channels, out_channels, 3, bn=True, padding=1),
            BasicConv2d(out_channels, out_channels, 3, bn=True, padding=1))
        self.training = training

    def forward(self, x):
        x = self.layer1(x)

        if self.training:
            x = x + torch.normal(mean=torch.full(x.size(),
                                                 fill_value=float(self.mean)),
                                 std=self.std).to(x.device)

        x = self.layer2(x)
        return x


if __name__ == '__main__':
    x = torch.randn((8, 58, 69, 73))
    net = Encoder(58)
    ret_for_od = net(x)
    print(ret_for_od.shape)
