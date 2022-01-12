import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from net.encoder import Encoder
from net.decoder import Decoder
from net.confidence import confidenceNetwork
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(comment='confidence')


class ConfidenceTrainer(nn.Module):
    def __init__(self, confidence_args, data_iter, device):
        super(ConfidenceTrainer, self).__init__()
        self.confidence = confidenceNetwork()
        self.init_params()
        self.data_iter = data_iter
        self.device = device

    def initialize(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.confidence.load_state_dict(checkpoint)

    def init_params(self):
        for param in self.confidence.parameters():
            if isinstance(param, nn.Conv2d):
                nn.init.xavier_uniform_(param.weight.data)
                nn.init.constant_(param.bias.data, 0.1)
            elif isinstance(param, nn.BatchNorm2d):
                param.weight.data.fill_(1)
                param.bias.data.zero_()
            elif isinstance(param, nn.Linear):
                param.weight.data.normal_(0, 0.01)
                param.bias.data.zero_()

    def forward(self, x):

        return self.confidence(x)

    def confidence_train(self,
                         epoch=5,
                         lr=0.0001,
                         weight_decay=0.99,
                         save_path='checkpoint/confidence2.pth'):
        optimizer = torch.optim.Adam(self.confidence.parameters(),
                                     lr,
                                     weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                         step_size=1000,
                                                         gamma=0.1)
        tb_log_intv = 100
        total_steps = 0
        for step in range(epoch):
            losses = []
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

                y_hat = F.softmax(y_hat, dim=1)
                #y_hat.requires_grad = True
                #print("y: ", y.shape, y_hat.shape,F.softmax(y_hat, dim=1).shape)
                optimizer.zero_grad()
                #loss = self.BCEloss(y_hat, y)
                loss = nn.MSELoss()(y_hat, y)
                loss.backward()
                optimizer.step()
                total_steps += 1
                losses.append(loss.item())
                if i % tb_log_intv == 0 and i != 0:
                    #print(nn.MSELoss()(y_hat, y).item())
                    avgl = np.mean(losses[-tb_log_intv:])
                    print("iter_Loss:", avgl)
                    writer.add_scalar("iter_Loss",
                                      avgl,
                                      global_step=total_steps)
                #TODO: 注意rain和temp的边界-99999判断，用一个mask记录-99999
            print('total_loss:{}'.format(np.mean(losses)))
            writer.add_scalar("epoch_Loss", np.mean(losses), global_step=step)
        writer.flush()
        torch.save(self.confidence.state_dict(), save_path)
        return

    def confidence_evaluate(self):
        self.confidence.eval()
        total_steps = 0
        losses = []
        for i, iter in enumerate(tqdm(self.data_iter)):
            [input, rain, temp] = iter
            input = input.type(torch.FloatTensor).to(self.device)
            rain = rain.type(torch.FloatTensor).to(self.device)
            rain = rain[:, 8:61, 8:65]  #这个地方还得再算算，边界判断！
            y_hat = self.confidence(input)
            rain = self.zero_mask(rain)
            y = torch.zeros_like(y_hat).to(self.device)
            for j in range(rain.shape[0]):
                #print(torch.sum(rain[j]))
                if torch.sum(rain[j]) > 302.1:
                    y[j] = torch.Tensor((0, 1))
                else:
                    y[j] = torch.Tensor((1, 0))
            with torch.no_grad():
                y_hat = F.softmax(y_hat, dim=1)
                loss = nn.MSELoss()(y_hat, y)
                total_steps += 1
                losses.append(loss.item())
        print(
            'total num: ',
            total_steps,
            " total MSEloss: {:.5f}".format(np.mean(losses)),
        )
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