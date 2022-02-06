import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_conv import BasicConv2d

class Conv2times(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 relu=True,
                 bn=True,
                 **kwargs):
        super(Conv2times, self).__init__()

        self.conv2times = nn.Sequential(BasicConv2d(in_channels, out_channels,
                                                    kernel_size, padding, relu, bn, **kwargs),
                                        BasicConv2d(out_channels, out_channels,
                                                    kernel_size, padding, relu, bn, **kwargs))

    def forward(self, x):
        return self.conv2times(x)


class Downsample(nn.Module):
    def __init__(self,
                 pooling_size,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 relu=True,
                 bn=True,
                 **kwargs):
        super(Downsample, self).__init__()
        self.pooling = nn.MaxPool2d(pooling_size)
        # self.conv1 = BasicConv2d(in_channels, out_channels, kernel_size, padding, relu, bn, **kwargs)
        # self.conv2 = BasicConv2d(out_channels, out_channels, kernel_size, padding, relu, bn, **kwargs)
        self.conv = Conv2times(in_channels, out_channels, kernel_size, padding, relu, bn, **kwargs)

    def forward(self, x):
        x = self.pooling(x)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self,
                 scale_factor,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 relu = True,
                 bn = True,
                 bilinear=True,
                 **kwargs):
        super(Upsample, self).__init__()

        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_channels, in_channels // 2, kernel_size, padding=padding))
        else:
            ##Transposed Conv##
            pass

        # self.conv1 = BasicConv2d(in_channels, out_channels, kernel_size, padding, relu, bn, **kwargs)
        # self.conv2 = BasicConv2d(out_channels, out_channels, kernel_size, padding, relu, bn, **kwargs)
        self.conv = Conv2times(in_channels, out_channels, kernel_size, padding, relu, bn, **kwargs)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1,
                   [diffX // 2, diffX - (diffX // 2),
                    diffY // 2, diffY - (diffY // 2)])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


if __name__ == '__main__':
    down = Downsample(2, 22, 44, 3, 1)
    up = Upsample(2, 44, 22, 3, 1)
    print(down)
    print(up)
    x = torch.rand(8, 22, 69, 73)
    xx = torch.rand(8, 22, 69, 73)
    t = down(x)
    x_hat = up(t, xx)
    print(t.shape)
    print(x_hat.shape)
