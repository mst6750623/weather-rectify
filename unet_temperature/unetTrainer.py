import torch
import torch.nn as nn
import numpy as np
from net.conv_unet import ConvUNet
from net.focal import FocalLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



class UNetTrainer(nn.Module):
    def __init__(self,
                 args,
                 train_iter,
                 evaluate_iter,
                 device,
                 writer='unet'):
        super(UNetTrainer, self).__init__()
        self.net = ConvUNet(args['in_channels'],
                            args['n_classes'])
        self.init_params()
        self.train_iter = train_iter
        self.evaluate_iter = evaluate_iter
        self.device = device
        self.writer = SummaryWriter(comment=writer)


    def initialize(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint)
        print('loading checkpoint:', checkpoint_path)

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

    def unet_train(self,
                      epoch,
                      lr,
                      save_path):
        optimizer = torch.optim.Adam(list(self.net.parameters()), lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                                  step_size=500000,
        #                                                  gamma=0.5)
        tb_log_intv = 200
        total_steps = 0
        evaluate_loss = 99999
        print('lr:', lr)
        print('total epoch:', epoch)
        for step in range(epoch):
            losses = []
            print('epoch: ', step)
            for i, iter in enumerate(tqdm(self.train_iter)):
                input, _, temperature = iter
                input = input.type(torch.FloatTensor).to(self.device)
                temperature = temperature.type(torch.FloatTensor).to(
                    self.device)  #(N, H, W)

                batch_size = input.shape[0]
                assert temperature.shape[0] == batch_size

                torch.set_printoptions(profile="full")

                pred_temperature = self.net(input)  #(N, H, W)

                with torch.no_grad():
                    mask = self.get_mask(temperature)
                    valid_points = torch.sum(mask)

                optimizer.zero_grad()
                loss = nn.L1Loss(reduction='sum')(mask * temperature,
                                                  mask * pred_temperature) / (batch_size * valid_points)
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
                    temp_evaluate_loss = self.unet_evaluate()
                    if temp_evaluate_loss < evaluate_loss:
                        evaluate_loss = temp_evaluate_loss
                        torch.save(self.net.state_dict(), save_path)
                    self.net.train()

            #每100个epoch存一次
            if step % 100 == 0:
                torch.save(self.net.state_dict(), save_path[:-4] + '_{}'.format(step) + '.pth')
            print('total_loss:{}'.format(np.mean(losses)))
            self.writer.add_scalar("epoch_Loss",
                                   np.mean(losses),
                                   global_step=step)
            #每个epoch都save
            # if step % 1 == 0:
            #     temp_evaluate_loss = self.unet_evaluate()
            #     if temp_evaluate_loss < evaluate_loss:
            #         evaluate_loss = temp_evaluate_loss
            #         torch.save(self.net.state_dict(), save_path)
            #     self.net.train()
        self.writer.flush()
        return

    def unet_evaluate(self):
        total_steps = 0
        losses = []
        self.net.eval()
        for i, iter in enumerate(tqdm(self.evaluate_iter)):
            with torch.no_grad():
                input, _, temperature = iter
                input = input.type(torch.FloatTensor).to(self.device)
                batch_size = temperature.shape[0]

                temperature = temperature.type(torch.FloatTensor).to(self.device)
                pred_temperature = self.net(input)

                mask = self.get_mask(temperature)
                valid_points = torch.sum(mask)

                loss = nn.L1Loss(reduction = 'sum')(mask * temperature,
                                                    mask * pred_temperature) / (batch_size * valid_points)

                total_steps += 1
                losses.append(loss.item())
        print(
            'Evaluate total num: ',
            total_steps,
            " total MAEloss: {:.5f}".format(np.mean(losses)),
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

