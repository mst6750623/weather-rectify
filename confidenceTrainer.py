import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from net.encoder import Encoder
from net.decoder import Decoder
from net.confidence import confidenceNetwork
from net.focal import FocalLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ConfidenceTrainer(nn.Module):
    def __init__(self, train_iter, evaluate_iter, device, writer='confidence'):
        super(ConfidenceTrainer, self).__init__()
        self.confidence = confidenceNetwork()
        self.init_params()
        self.train_iter = train_iter
        self.evaluate_iter = evaluate_iter
        print(len(self.train_iter), len(self.evaluate_iter))
        self.device = device
        self.writer = SummaryWriter(comment=writer)

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
                         save_path='checkpoint/confidence.pth'):
        optimizer = torch.optim.Adam(self.confidence.parameters(), lr)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.1)
        tb_log_intv = 100
        total_steps = 0
        evaluate_loss = 99999
        for step in range(epoch):
            losses = []
            print('epoch: {step}')
            for i, iter in enumerate(tqdm(self.train_iter)):
                input, rain, temp, time = iter
                #原本是double的，但网络参数是float，不改输入的话，就得在网络参数上手动改（较为麻烦）
                input = input.type(torch.FloatTensor).to(self.device)

                if rain == -99999:
                    continue
                elif rain < 0.1:
                    y = torch.Tensor([1, 0]).to(self.device)
                else:
                    y = torch.Tensor([0, 1]).to(self.device)

                y_hat = self.confidence(input)
                y_hat = F.softmax(y_hat, dim=1)
                #y_hat.requires_grad = True
                #print("y: ", y.shape, y_hat.shape,F.softmax(y_hat, dim=1).shape)
                optimizer.zero_grad()
                #loss = self.BCEloss(y_hat, y)
                loss = self.FocalLoss(y_hat, y)
                loss.backward()
                optimizer.step()
                total_steps += 1
                losses.append(loss.item())
                if i % tb_log_intv == 0 and i != 0:
                    #print(nn.MSELoss()(y_hat, y).item())
                    avgl = np.mean(losses[-tb_log_intv:])
                    print("iter_Loss:", avgl)
                    self.writer.add_scalar("iter_Loss",
                                           avgl,
                                           global_step=total_steps)
                #TODO: 注意rain和temp的边界-99999判断，用一个mask记录-99999
            print('total_loss:{}'.format(np.mean(losses)))
            self.writer.add_scalar("epoch_Loss",
                                   np.mean(losses),
                                   global_step=step)
            #每个epoch都save
            if step % 1 == 0:
                temp_evaluate_loss = self.confidence_evaluate()
                if temp_evaluate_loss < evaluate_loss:
                    evaluate_loss = temp_evaluate_loss
                    torch.save(self.confidence.state_dict(), save_path)

        self.writer.flush()
        #torch.save(self.confidence.state_dict(), save_path)
        return

    def confidence_evaluate(self):
        total_steps = 0
        losses = []
        for i, iter in enumerate(tqdm(self.evaluate_iter)):
            input, rain, temp = iter
            input = input.type(torch.FloatTensor).to(self.device)
            if rain == -99999:
                continue
            elif rain < 0.1:
                y = torch.Tensor([1, 0]).to(self.device)
            else:
                y = torch.Tensor([0, 1]).to(self.device)

            y_hat = self.confidence(input)
            y_hat = F.softmax(y_hat, dim=1)

            with torch.no_grad():
                y_hat = F.softmax(y_hat, dim=1)
                loss = self.FocalLoss(y_hat, y)
                total_steps += 1
                losses.append(loss.item())
        print(
            'total num: ',
            total_steps,
            " total MSEloss: {:.5f}".format(np.mean(losses)),
        )
        return np.mean(losses)

    def BCEloss(self, x, target, reduction='mean'):
        return nn.BCELoss(reduction=reduction)(x, target)

    def FocalLoss(self, x, target):
        focal = FocalLoss(2, alpha=0.25, gamma=2).to(self.device)
        return focal(x, target)

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