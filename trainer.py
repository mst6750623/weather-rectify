import torch
import torch.nn as nn
from net.encoder import encoder
from net.decoder import decoder
from net.confidence import confidenceNetwork


class Trainer():
    def __init__(self, args):
        self.encoder = encoder(args.in_channels,
                               args.mid_channels,
                               args.out_channels,
                               mean=0,
                               std=1e-1)
        self.decoder = decoder(args.in_channels)
