import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from net.encoder import Encoder
from net.decoder import Decoder
from net.confidence import confidenceNetwork
from tqdm import tqdm


class Trainer(nn.Module):
    def __init__(self, encoder_args, decoder_args, confidence_args, data_iter,
                 device):
        super(Trainer, self).__init__()
        self.encoder = Encoder(encoder_args['in_channels'],
                               encoder_args['mid_channels'],
                               encoder_args['out_channels'],
                               mean=0,
                               std=1e-1)
        self.decoder = Decoder(
            decoder_args['in_channels'],
            decoder_args['mid_channels'],
            decoder_args['out_channels'],
        )
        self.confidence = confidenceNetwork()
        self.data_iter = data_iter
        self.device = device

    def forward(self, x):
        '''print('encoder:', self.encoder)
        print('decoder:', self.decoder)
        print('confidence:', self.confidence)'''
        y = self.encoder(x)
        return y, self.confidence(x), self.decoder(y)

    def confidence_train(self,
                         epoch=10,
                         lr=0.0001,
                         weight_decay=0.99,
                         save_path='checkpoint/confidence.pth'):
        optimizer = torch.optim.Adam(self.confidence.parameters(),
                                     lr,
                                     weight_decay=weight_decay)

        for step in range(epoch):
            for i, iter in enumerate(tqdm(self.data_iter)):
                [input, rain, temp] = iter
                #原本是double的，但网络参数是float，不改输入的话，就得在网络参数上手动改（较为麻烦）
                input = input.type(torch.FloatTensor).to(self.device)
                rain = rain.type(torch.FloatTensor).to(self.device)
                rain = rain[:, 8:61, 8:65]  #这个地方不要还得再算算，边界判断！
                y_hat = self.confidence(input)
                rain = self.zero_mask(rain)
                y = torch.zeros_like(y_hat).to(self.device)
                for j in range(rain.shape[0]):
                    #print(torch.sum(rain[j]))
                    if torch.sum(rain[j]) > 302.1:
                        y[j] = torch.Tensor((0, 1))
                    else:
                        y[j] = torch.Tensor((1, 0))
                #print(y_hat, y_hat.shape)

                y_hat = F.softmax(y_hat, dim=1).to(self.device)
                #y_hat.requires_grad = True
                #print("y: ", y.shape, y_hat.shape,F.softmax(y_hat, dim=1).shape)
                optimizer.zero_grad()
                loss = self.BCEloss(y_hat, y)
                if i % 100 == 0 and i != 0:
                    print("loss:", loss)
                loss.backward()
                optimizer.step()
                #TODO: 注意rain和temp的边界-99999判断，用一个mask记录-99999
                #算算rain的分布，temp的分布

        torch.save(self.confidence.state_dict(), save_path)
        return

    def BCEloss(self, x, target, reduction='mean'):
        return nn.BCELoss(reduction=reduction)(x, target)

    #将所有的-99999变成0
    def zero_mask(self, x):
        zero = torch.zeros_like(x)
        return torch.where(x == -99999, zero, x)

    def generateOneHot(self, softmax):
        maxIdxs = torch.argmax(softmax, dim=1, keepdim=True).cpu().long()
        oneHotMask = torch.zeros(softmax.shape, dtype=torch.float32)
        oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
        #oneHotMask = oneHotMask.unsqueeze(-2)
        return oneHotMask