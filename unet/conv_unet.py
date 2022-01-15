import torch
import torch.nn as nn
from basic_conv import BasicConv2d
from net_parts import *


class ConvUNet(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True, **kwargs):
        super(ConvUNet, self).__init__()

        self.input_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        ##后续可以改结构，但必须保证：
        #down和up对称
        #down的in_channels与out_channels之间必须是2倍关系
        self.in_conv = Conv2times(in_channels, 64, 3, 1)

        self.down1 = Downsample(2, 64, 128, 3, 1)
        self.down2 = Downsample(2, 128, 256, 3, 1)
        self.down3 = Downsample(2, 256, 512, 3, 1)
        self.down4 = Downsample(2, 512, 1024, 3, 1)

        self.up1 = Upsample(2, 1024, 512, 3, 1, bilinear)
        self.up2 = Upsample(2, 512, 256, 3, 1, bilinear)
        self.up3 = Upsample(2, 256, 128, 3, 1, bilinear)
        self.up4 = Upsample(2, 128, 64, 3, 1, bilinear)

        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)


if __name__ == '__main__':
    net = ConvUNet(22, 1)
    x = torch.rand(8, 22, 69, 73)
    t = net(x)
    print(t.shape)