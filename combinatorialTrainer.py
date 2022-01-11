import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from net.CombinatorialNetwork import CombinatorialNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class CombinatorialTrainer(nn.Module):
    def __init__(self, combinatorial_args, data_iter, device):
        super(CombinatorialTrainer, self).__init__()
        self.net = CombinatorialNet(
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
        self.init_params()
        self.data_iter = data_iter
        self.device = device

    def init_params(self):
        for param in self.net.parameters():
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

        return self.net(x)

    def encoder_train(self,
                      epoch=100,
                      lr=100,
                      save_path1='checkpoint/encoder.pth',
                      save_path2='checkpoint/decoder.pth'):
        optimizer = torch.optim.Adam(
            list(self.net.encoder.parameters()) +
            list(self.net.decoder.parameters()), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                         step_size=1000,
                                                         gamma=0.1)
        tb_log_intv = 100
        total_steps = 0
        for step in range(epoch):
            losses = []
            for i, iter in enumerate(tqdm(self.data_iter)):
                [input, _, _] = iter
                input = input.type(torch.FloatTensor).to(self.device)
                torch.set_printoptions(profile="full")

                y_hat = self.net(input, isOrdinal=False)
                #print(y_hat[0][0])
                with torch.no_grad():
                    mask = self.get_mask(input)

                optimizer.zero_grad()
                loss = nn.MSELoss()(y_hat * mask, input * mask)
                loss.backward()
                optimizer.step()
                total_steps += 1
                losses.append(loss.item())
                if i % tb_log_intv == 0 and i != 0:
                    avgl = np.mean(losses[-tb_log_intv:])
                    print("iter_Loss:", avgl)
                    writer.add_scalar("iter_Loss",
                                      avgl,
                                      global_step=total_steps)

                #TODO: 注意rain和temp的边界-99999判断，用一个mask记录-99999
            print('total_loss:{}'.format(np.mean(losses)))
            writer.add_scalar("epoch_Loss",
                              np.mean(losses),
                              global_step=total_steps)
        writer.flush()
        torch.save(self.net.encoder.state_dict(), save_path1)
        torch.save(self.net.decoder.state_dict(), save_path2)
        return

    def BCEloss(self, x, target, reduction='mean'):
        return nn.BCELoss(reduction=reduction)(x, target)

    #生成将所有的-99999变成0,其他为1的mask
    def get_mask(self, x):
        zero = torch.zeros_like(x)
        ones = torch.ones_like(x)

        x = torch.where(x > -99999, ones, x)
        x = torch.where(x == -99999, zero, x)
        return x

    def generateOneHot(self, softmax):
        maxIdxs = torch.argmax(softmax, dim=1, keepdim=True).cpu().long()
        oneHotMask = torch.zeros(softmax.shape, dtype=torch.float32)
        oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
        #oneHotMask = oneHotMask.unsqueeze(-2)
        return oneHotMask