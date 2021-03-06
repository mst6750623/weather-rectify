from hashlib import new
from tkinter import E
import torch
import torch.nn as nn
import numpy as np
from net.conv_unet_original import ConvUNet
from torch.utils.data.dataloader import DataLoader
from dataset import gridDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from net.timeReflect import TimeReflect


class model(nn.Module):
    def __init__(self, train_data, **kwargs):
        super(model, self).__init__()
        config = kwargs['config']
        args = config['unet']
        self.net = ConvUNet(args['in_channels'], args['n_classes'])
        self.net.train()
        self.timeReflect = TimeReflect(8, 4)
        self.timeReflect.train()
        self.init_params()
        # 若设置is FirstTime=True,则将会花一些时间对训练输入数据扫描计算mean和std，进行归一化
        train_dataset = gridDataset(train_data,
                                    isTrain=True,
                                    isFirstTime=False,
                                    nwp_num=args['in_channels'])
        evaluate_dataset = gridDataset(train_data,
                                       isTrain=False,
                                       isFirstTime=False,
                                       nwp_num=args['in_channels'])
        self.train_iter = DataLoader(train_dataset,
                                     batch_size=config['batch_size'],
                                     num_workers=config['num_workers'],
                                     shuffle=True,
                                     pin_memory=True)
        self.evaluate_iter = DataLoader(evaluate_dataset,
                                        batch_size=32,
                                        shuffle=False,
                                        pin_memory=True)

        self.device = kwargs['device']
        self.writer = SummaryWriter(comment='time')
        self.cof = 1
        self.max_epochs = kwargs['max_epochs']
        self.show_trainloss_every_num_iterations_per_epoch = kwargs[
            'show_trainloss_every_num_iterations_per_epoch']
        self.show_validperformance_every_num_epochs = kwargs[
            'show_validperformance_every_num_epochs']
        self.valid_example = train_dataset.length
        self.train_loss = 'nn.L1Loss'
        self.valid_perfomace = 'nn.L1Loss'

    def initialize(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['unet_state_dict'])
        self.timeReflect.load_state_dict(checkpoint['time_reflect_state_dict'])
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

    def train(self, lr=1e-5, save_path='checkpoint/unet_out.pth'):
        epoch = self.max_epochs
        optimizer = torch.optim.Adam(list(self.net.parameters()), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                         step_size=10000,
                                                         gamma=0.5)
        tb_log_intv = self.show_trainloss_every_num_iterations_per_epoch
        total_steps = 0
        evaluate_loss = 99999

        for step in range(epoch):
            losses = []
            print('epoch: ', step)
            for i, iter in enumerate(tqdm(self.train_iter)):
                input, _, temperature, time = iter
                input = input.type(torch.FloatTensor).to(self.device)
                temperature = temperature.type(torch.FloatTensor).to(
                    self.device)  #(N, H, W)
                time = time.to(self.device)
                time = self.timeReflect(time)

                batch_size = input.shape[0]
                time = time.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 69, 73)

                torch.set_printoptions(profile="full")

                with torch.no_grad():
                    temperature[temperature < -10.0] = -99999.0
                    mask = self.get_mask(temperature)
                    valid_points = torch.sum(mask)
                    #time = self.generateLabelOneHot(time, shape = (time.shape[0], 8))
                    #time = time.squeeze()
                newinput = torch.cat([input, time], dim=1)

                pred_temperature = self.net(newinput)  # (N, H, W)
                #loss_time = nn.CrossEntropyLoss()(time_hat, time)
                optimizer.zero_grad()
                loss = nn.L1Loss(reduction='mean')(
                    mask * temperature, mask * pred_temperature) / valid_points
                #total_loss = 10000 * loss + self.cof * loss_time
                total_loss = loss
                total_loss.backward()
                optimizer.step()
                self.scheduler.step()
                total_steps += 1
                losses.append(loss.item())

                if i % tb_log_intv == 0 and i != 0:
                    avgl = np.mean(losses[-tb_log_intv:])
                    print("iter_Loss:", avgl)
                    self.writer.add_scalar("iter_Loss",
                                           total_loss.item(),
                                           global_step=total_steps)
                    #self.writer.add_scalar("time_Loss",loss_time.item(),global_step=total_steps)
                    self.writer.add_scalar("classification_Loss",
                                           loss.item(),
                                           global_step=total_steps)
                    print(optimizer.state_dict()['param_groups'][0]['lr'])

            if step % self.show_validperformance_every_num_epochs == 0 and step != 0:
                #torch.save(self.net.state_dict(),save_path[:-4] + '_{}'.format(step) + '.pth')
                temp_evaluate_loss = self.unet_evaluate()
                self.writer.add_scalar("evaluate_Loss",
                                       temp_evaluate_loss.item(),
                                       global_step=total_steps)
                if temp_evaluate_loss < evaluate_loss:
                    evaluate_loss = temp_evaluate_loss

                    torch.save(
                        {
                            'time_reflect_state_dict':
                            self.timeReflect.state_dict(),
                            'unet_state_dict':
                            self.net.state_dict(),
                        }, save_path[:-4] + '_best.pth')
                self.net.train()
            #print('total_loss:{}'.format(np.mean(losses)))
            self.writer.add_scalar("epoch_Loss",
                                   np.mean(losses),
                                   global_step=step)

        self.writer.flush()
        return

    def unet_evaluate(self):
        total_steps = 0
        losses = []
        self.net.eval()
        for i, iter in enumerate(tqdm(self.evaluate_iter)):
            with torch.no_grad():
                input, _, temperature, time = iter
                input = input.type(torch.FloatTensor).to(self.device)
                time = time.to(self.device)
                batch_size = temperature.shape[0]
                time = self.timeReflect(time)
                time = time.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 69, 73)
                #print(time.shape)
                newinput = torch.cat([input, time], dim=1)
                temperature = temperature.type(torch.FloatTensor).to(
                    self.device)
                pred_temperature = self.net(newinput)

                mask = self.get_mask(temperature)
                valid_points = torch.sum(mask)

                loss = nn.L1Loss(reduction='sum')(
                    mask * temperature, mask * pred_temperature) / valid_points

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

    def generateLabelOneHot(self, x, shape=(32, 8)):
        result = torch.zeros(shape).to(self.device)
        for idx, item in enumerate(x):
            result[idx][int(item)] = 1
        #print(x, result)
        return result

    def generateOneHot(self, softmax):
        maxIdxs = torch.argmax(softmax, dim=1, keepdim=True).cpu().long()
        oneHotMask = torch.zeros(softmax.shape, dtype=torch.float32)
        oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
        #oneHotMask = oneHotMask.unsqueeze(-2)
        return oneHotMask
