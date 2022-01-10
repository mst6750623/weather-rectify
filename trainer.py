import torch
import torch.nn as nn
from net.encoder import encoder
from net.decoder import decoder
from net.confidence import confidenceNetwork


class Trainer(nn.Module):
    def __init__(self, encoder_args, decoder_args, confidence_args):
        super(Trainer, self).__init__()
        self.encoder = encoder(encoder_args['in_channels'],
                               encoder_args['mid_channels'],
                               encoder_args['out_channels'],
                               mean=0,
                               std=1e-1)
        self.decoder = decoder(
            decoder_args['in_channels'],
            decoder_args['mid_channels'],
            decoder_args['out_channels'],
        )
        self.confidence = confidenceNetwork()

    def forward(self, x):
        '''print('encoder:', self.encoder)
        print('decoder:', self.decoder)
        print('confidence:', self.confidence)'''
        return self.encoder(x)
