import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .OrdinalRegression import OrdinalRegressionNetwork as Orn


class CombinatorialNet(nn.Module):
    def __init__(self,
                 in_channels_for_enc,
                 mid_channels_for_enc,
                 out_channels_for_enc,
                 mid_channels_for_OD,
                 out_channels_for_OD,
                 mid_channels_for_dec,
                 out_channels_for_dec,
                 num_of_classes,
                 noise_mean=0,
                 noise_std=1e-1):
        super(CombinatorialNet, self).__init__()

        self.encoder = Encoder(in_channels_for_enc, mid_channels_for_enc,
                               out_channels_for_enc, noise_mean, noise_std)
        self.decoder = Decoder(out_channels_for_enc, mid_channels_for_dec,
                               out_channels_for_dec)
        self.OD = Orn(out_channels_for_enc, mid_channels_for_OD,
                      out_channels_for_OD, num_of_classes)

    def forward(self, x, isOrdinal):
        x = self.encoder(x)
        if isOrdinal:
            x = self.OD(x)
        else:
            x = self.decoder(x)
        return x


if __name__ == '__main__':
    x = torch.randn((8, 58, 69, 73))
    net = CombinatorialNet(58, 32, 64, 128, 32, 32, 58, 5)  ##4类好像就够了吧？
    ret_for_od, ret_for_dec = net(x)
    print(ret_for_od.shape)
    print(ret_for_dec.shape)
