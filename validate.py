import torch
import torch.nn as nn
import numpy as np
from net.CombinatorialNetwork import CombinatorialNet
from net.confidence import confidenceNetwork


class Validate(nn.Module):
    def __init__(self, combinatorial_args, data_iter, device):
        super(Validate, self).__init__()
        self.confidence = confidenceNetwork()
        self.prediction = CombinatorialNet(
            combinatorial_args['encoder']['in_channels'],
            combinatorial_args['encoder']['mid_channels'],
            combinatorial_args['encoder']['out_channels'],
            combinatorial_args['ordinal']['mid_channels'],
            combinatorial_args['ordinal']['out_channels'],
            combinatorial_args['decoder']['mid_channels'],
            combinatorial_args['decoder']['out_channels'],
            combinatorial_args['nclass'],
            noise_mean=0,
            noise_std=1e-1)
        self.data_iter = data_iter
        self.device = device

    def initialize(self, confidence_path, encoder_path, ordinal_path):
        confidence_ckpt = torch.load(confidence_path)
        encoder_ckpt = torch.load(encoder_path)
        ordinal_ckpt = torch.load(ordinal_path)
        self.confidence.load_state_dict(confidence_ckpt)
        self.confidence.eval()
        self.prediction.encoder.load_state_dict(encoder_ckpt)
        self.prediction.OD.load_state_dict(ordinal_ckpt)
        self.prediction.eval()

    def forward(self):
        for features, locs in self.data_iter:
            features = features.to(self.device)
            locs = locs.to(self.device)

            with torch.no_grad():
                encoder, _ = self.autoencoder(features)
                predictValues = self.regressor(encoder)
                rainPreds = self.classification(features)

                rainPredsSoftMax = F.softmax(rainPreds, dim=1)
                rainOnehot = self.generateOneHot(rainPredsSoftMax)

                regressionValues = 0.5 * (torch.sum(
                    (predictValues > 0.5).float(), dim=1).view(-1, 1))
                zeros = torch.zeros(regressionValues.size()).to(self.device)

                regressionValues = torch.matmul(
                    rainOnehot,
                    torch.cat([zeros, regressionValues],
                              dim=1).unsqueeze(-1)).squeeze(-1)

                regressionValues = regressionValues.item()
                print(locs)
                print(regressionValues)