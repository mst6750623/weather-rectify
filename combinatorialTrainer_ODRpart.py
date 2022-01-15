import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from net.CombinatorialNetwork import CombinatorialNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from net.OrdinalFocalLoss import ordinalLoss


class ODRCombinatorialTrainer(nn.Module):
    def __init__(self,
                 combinatorial_args,
                 train_iter,
                 evaluate_iter,
                 device,
                 writer='odr'):
        super(ODRCombinatorialTrainer, self).__init__()
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
        self.alpha = 1
        self.beta = 1
        #self.threshold = torch.tensor([20.0, 10.0, 3.0, 0.1]).to(device=device)
        self.threshold = torch.tensor([0.1, 3.0, 10.0, 20.0]).to(device=device)

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
                      lr=0.0001,
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
                    self.writer.add_scalar("iter_Loss",
                                           avgl,
                                           global_step=total_steps)

            print('total_loss:{}'.format(np.mean(losses)))
            self.writer.add_scalar("epoch_Loss",
                                   np.mean(losses),
                                   global_step=step)
        self.writer.flush()
        torch.save(self.net.encoder.state_dict(), save_path1)
        torch.save(self.net.decoder.state_dict(), save_path2)
        return

    #####trainer for Ordinal Distribution Regression#######
    def odr_train(self,
                  epoch=100,
                  lr=0.0001,
                  encoder_path='checkpoint/encoder.pth',
                  save_path1='checkpoint/encoderwithodr.pth',
                  save_path2='checkpoint/odr.pth'):
        if os.path.exists(encoder_path):
            encoder_ckpt = torch.load(encoder_path)
            self.net.encoder.load_state_dict(encoder_ckpt)
            print('encoder initialize success!')
        else:
            print('Train from scratch!')
        optimizer = torch.optim.Adam(
            list(self.net.OD.parameters()) +
            list(self.net.encoder.parameters()), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                         step_size=50000,
                                                         gamma=0.5)
        tb_log_intv = 100
        total_steps = 0
        evaluate_loss = 99999
        for step in range(epoch):
            losses = []
            for i, iter in enumerate(tqdm(self.train_iter)):
                input, rain, _, _ = iter
                input = input.type(torch.FloatTensor).to(self.device)
                rain = rain.type(torch.FloatTensor).to(
                    self.device)  #rain：(N, 1)
                torch.set_printoptions(profile="full")

                able_list = []
                rain_resample = torch.Tensor().to(self.device)
                for j in range(rain.shape[0]):
                    #print(torch.sum(rain[j]))
                    if rain[j] >= 0.1:
                        rain_resample = torch.concat(
                            (rain_resample, torch.Tensor([rain[j]
                                                          ]).to(self.device)))
                        able_list.append(j)
                    else:
                        # 重采样5%几率采样
                        if torch.rand(1) > 0.95:
                            rain_resample = torch.concat(
                                (rain_resample,
                                 torch.Tensor([0]).to(self.device)))
                            able_list.append(j)
                able_list = torch.Tensor(able_list).int().to(self.device)

                input = torch.index_select(input, dim=0, index=able_list)
                if input.shape[0] == 0:
                    continue
                y_hat = self.net(input, isOrdinal=True)  #y_hat is (N, 4)

                #先算出目标
                #TODO: -99999怎么处理? 改focal, 等待unet验证是否可行……
                #REPO:现在先按0处理
                y = torch.zeros((rain_resample.shape[0],
                                 self.threshold.shape[0])).to(self.device)

                for j in range(rain_resample.shape[0]):
                    for k in range(self.threshold.shape[0]):
                        if rain_resample[j] > self.threshold[k]:
                            y[j][k] = 1
                        else:
                            break

                ###no need for grad of encoder###
                #self.net.encoder.weight.requires_grad = False
                # for param in self.net.encoder.parameters():
                #     param.requires_grad = False

                optimizer.zero_grad()

                ce, emd = ordinalLoss()(y_hat, y)
                #print(ce, emd)
                loss = self.alpha * ce + self.beta * emd
                loss.backward()
                optimizer.step()
                total_steps += 1
                losses.append(loss.item())
                if i % tb_log_intv == 0 and i != 0:
                    avgl = np.mean(losses[-tb_log_intv:])
                    #print("iter_Loss:", avgl)
                    self.writer.add_scalar("iter_Loss",
                                           loss.item(),
                                           global_step=total_steps)
                if i % 500000 == 0 and i != 0:
                    temp_evaluate_loss = self.odr_evaluate()
                    if temp_evaluate_loss < evaluate_loss:
                        evaluate_loss = temp_evaluate_loss
                        torch.save(self.net.encoder.state_dict(), save_path1)
                        torch.save(self.net.OD.state_dict(), save_path2)
                    self.net.train()
            print('total_loss:{}'.format(np.mean(losses)))
            self.writer.add_scalar("epoch_Loss",
                                   np.mean(losses),
                                   global_step=step)
            if step % 1 == 0:
                temp_evaluate_loss = self.odr_evaluate()
                if temp_evaluate_loss < evaluate_loss:
                    evaluate_loss = temp_evaluate_loss
                    torch.save(self.net.encoder.state_dict(), save_path1)
                    torch.save(self.net.OD.state_dict(), save_path2)
                self.net.train()
        self.writer.flush()
        torch.save(self.net.encoder.state_dict(), save_path1)
        torch.save(self.net.OD.state_dict(), save_path2)
        return

    def odr_evaluate(self):
        total_steps = 0
        losses = []
        for i, iter in enumerate(tqdm(self.evaluate_iter)):
            with torch.no_grad():
                input, rain, _, _ = iter
                input = input.type(torch.FloatTensor).to(self.device)
                rain = rain.type(torch.FloatTensor).to(
                    self.device)  #rain：(N, 1)
                torch.set_printoptions(profile="full")
                y_hat = self.net(input, isOrdinal=True)  #y_hat is (N, 4)

                #先算出目标
                y = torch.zeros(
                    (rain.shape[0], self.threshold.shape[0])).to('cuda')
                for j in range(rain.shape[0]):
                    for k in range(self.threshold.shape[0]):
                        if rain[j] > self.threshold[k]:
                            y[j][k] = 1
                        else:
                            break

                ce, emd = ordinalLoss()(y_hat, y)
                loss = self.alpha * ce + self.beta * emd
                total_steps += 1
                losses.append(loss.item())

        print(
            'total num: ',
            total_steps,
            " total ordinalloss: {:.5f}".format(np.mean(losses)),
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