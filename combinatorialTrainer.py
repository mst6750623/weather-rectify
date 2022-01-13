import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from net.CombinatorialNetwork import CombinatorialNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from dataset import gridDataset
import yaml


class CombinatorialTrainer(nn.Module):
    def __init__(self,
                 combinatorial_args,
                 train_iter,
                 evaluate_iter,
                 device,
                 writer='encoder-decoder'):
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
        self.train_iter = train_iter
        self.evaluate_iter = evaluate_iter
        self.device = device
        self.writer = SummaryWriter(comment=writer)

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
                      lr=0.1,
                      save_path1='checkpoint/encoder.pth',
                      save_path2='checkpoint/decoder.pth'):
        optimizer = torch.optim.Adam(
            list(self.net.encoder.parameters()) +
            list(self.net.decoder.parameters()), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                         step_size=179000,
                                                         gamma=0.1)
        tb_log_intv = 200
        total_steps = 0
        evaluate_loss = 99999

        for step in range(epoch):
            losses = []
            print('epoch: ', step)
            for i, iter in enumerate(tqdm(self.train_iter)):
                input, _, _, _ = iter
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
                self.writer.add_scalar("iter_Loss",
                                       loss.item(),
                                       global_step=total_steps)
                '''if i % tb_log_intv == 0 and i != 0:
                    avgl = np.mean(losses[-tb_log_intv:])
                    print("iter_Loss:", avgl)'''

                #TODO: 注意rain和temp的边界-99999判断，用一个mask记录-99999 :tick
            print('total_loss:{}'.format(np.mean(losses)))
            self.writer.add_scalar("epoch_Loss",
                                   np.mean(losses),
                                   global_step=step)
            #每个epoch都save
            if step % 1 == 0:
                temp_evaluate_loss = self.combinatorial_evaluate()
                if temp_evaluate_loss < evaluate_loss:
                    evaluate_loss = temp_evaluate_loss
                    torch.save(self.net.encoder.state_dict(), save_path1)
                    torch.save(self.net.decoder.state_dict(), save_path2)
        self.writer.flush()
        #torch.save(self.net.encoder.state_dict(), save_path1)
        #torch.save(self.net.decoder.state_dict(), save_path2)
        return

    def combinatorial_evaluate(self):
        total_steps = 0
        losses = []
        for i, iter in enumerate(tqdm(self.evaluate_iter)):
            with torch.no_grad():
                input, rain, temp = iter
                input = input.type(torch.FloatTensor).to(self.device)
                y_hat = self.net(input, isOrdinal=False)
                mask = self.get_mask(input)

                y_hat = F.softmax(y_hat, dim=1)
                loss = nn.MSELoss()(y_hat * mask, input * mask)
                total_steps += 1
                losses.append(loss.item())
        print(
            'Evaluate total num: ',
            total_steps,
            " total MSEloss: {:.5f}".format(np.mean(losses)),
        )
        return np.mean(losses)

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


if __name__ == '__main__':
    device = 'cuda'
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    dataset = gridDataset(config['train_dir'], isTrain=True)
    data_iter = DataLoader(dataset,
                           batch_size=config['batch_size'],
                           num_workers=config['num_workers'],
                           shuffle=True)
    mean = 0
    std_tensor = torch.load('processed_data/std.pth')
    print(std_tensor.shape)
    std = 1
    losses = []
    tb_log_intv = 100
    needed_tensor = torch.zeros((1, 58, 69, 73))
    for i in range(58):
        needed_tensor[0][i] = torch.full((69, 73),
                                         fill_value=float(std_tensor[i] / 100))

    needed_tensor = needed_tensor.repeat((16, 1, 1, 1))
    print(needed_tensor.shape)
    for i, iter in enumerate(tqdm(data_iter)):
        [input, _, _] = iter
        input = input.type(torch.FloatTensor).to(device)

        random = input + torch.normal(mean=torch.full(input.size(),
                                                      fill_value=float(mean)),
                                      std=needed_tensor).to(device)
        loss = nn.MSELoss()(random, input)
        losses.append(loss.item())
        if i % tb_log_intv == 0 and i != 0:
            avgl = np.mean(losses[-tb_log_intv:])
            print("iter_Loss:", avgl)
