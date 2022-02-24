import torch
import torch.nn as nn

from .net_parts_spatial import *


class ConvUNet(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True, **kwargs):
        super(ConvUNet, self).__init__()

        self.input_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        ##后续可以改结构，但必须保证：
        #down和up对称
        #down的in_channels与out_channels之间必须是2倍关系
        self.coeff = 4
        self.in_conv = Conv2times(in_channels, 64 // self.coeff, 3, 1)

        self.down1 = Downsample(2,
                                64 // self.coeff,
                                128 // self.coeff,
                                3,
                                1,
                                H=34,
                                W=36)
        self.down2 = Downsample(2,
                                128 // self.coeff,
                                256 // self.coeff,
                                3,
                                1,
                                H=17,
                                W=18)
        self.down3 = Downsample(2,
                                256 // self.coeff,
                                512 // self.coeff,
                                3,
                                1,
                                H=8,
                                W=9)
        self.down4 = Downsample(2,
                                512 // self.coeff,
                                1024 // self.coeff,
                                3,
                                1,
                                H=4,
                                W=4)

        self.up1 = Upsample(2,
                            1024 // self.coeff,
                            512 // self.coeff,
                            3,
                            1,
                            bilinear,
                            H=8,
                            W=9)
        self.up2 = Upsample(2,
                            512 // self.coeff,
                            256 // self.coeff,
                            3,
                            1,
                            bilinear,
                            H=17,
                            W=18)
        self.up3 = Upsample(2,
                            256 // self.coeff,
                            128 // self.coeff,
                            3,
                            1,
                            bilinear,
                            H=34,
                            W=36)
        self.up4 = Upsample(2,
                            128 // self.coeff,
                            64 // self.coeff,
                            3,
                            1,
                            bilinear,
                            H=69,
                            W=73)

        self.out_conv = nn.Conv2d(64 // self.coeff, n_classes, kernel_size=1)

        '''self.timenet = nn.Sequential(
            nn.Flatten(), nn.Linear(64 // self.coeff * 69 * 73, 1024),
            nn.ReLU(True), nn.Linear(1024, 8))'''

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

        x = self.out_conv(x)

        #输出层仅仅是1 * 1卷积就输出了，没有激活函数
        return x.squeeze()


if __name__ == '__main__':
    net = ConvUNet(58, 1).to('cuda')
    x = torch.rand(8, 58, 69, 73).to('cuda')
    x, time = net(x)
    print(x.shape, time.shape)