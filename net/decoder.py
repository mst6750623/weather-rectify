import torch
import torch.nn as nn
from .confidence import BasicConv2d


class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Decoder, self).__init__()

        self.layer1 = BasicConv2d(in_channels,
                                  mid_channels,
                                  3,
                                  bn=True,
                                  padding=1)
        self.upsample = nn.Upsample(size=(17, 17), mode='bilinear')
        self.layer2 = nn.Sequential(
            BasicConv2d(mid_channels, mid_channels, 3, bn=True, padding=1),
            #BasicConv2d(mid_channels, out_channels, 3, bn=True, padding=1),
            ##经过实验，这里最后一层得加个padding才能还原为58*69*73，不加padding就是58*67*71
            ##我sb了，之前参数写错了,应该用1*1 conv
            BasicConv2d(mid_channels, out_channels, 1, bn = True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.upsample(x)
        x = self.layer2(x)
        return x


if __name__ == '__main__':
    x = torch.rand((8, 64, 8, 8))
    dec = Decoder(64, 32, 22)
    print(dec(x).shape)
