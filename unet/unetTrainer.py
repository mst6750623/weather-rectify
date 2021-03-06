import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from net.conv_unet import ConvUNet
from net.focal import FocalLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def regression_value(self, x):
    # to be continued
    return None


class UNetTrainer(nn.Module):
    def __init__(self, args, train_iter, evaluate_iter, device, writer='unet'):
        super(UNetTrainer, self).__init__()
        self.net = ConvUNet(args['in_channels'], args['n_classes'])
        self.init_params()
        self.train_iter = train_iter
        self.evaluate_iter = evaluate_iter
        self.device = device
        self.writer = SummaryWriter(comment=writer)
        self.threshold = torch.tensor([0.1, 3.0, 10.0, 20.0]).to(device=device)

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

    def unet_train(self, epoch, lr, save_path):
        optimizer = torch.optim.Adam(list(self.net.parameters()), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                         step_size=150000,
                                                         gamma=0.1)
        tb_log_intv = 200
        total_steps = 0
        evaluate_loss = 99999
        print('lr:', lr)
        print('total epoch:', epoch)
        for step in range(epoch):
            losses = []
            print('epoch: ', step)
            for i, iter in enumerate(tqdm(self.train_iter)):
                input, rain, _, time = iter
                input = input.type(torch.FloatTensor).to(self.device)
                rain = rain.type(torch.FloatTensor).to(self.device)
                assert len(rain.shape) == 3  #??????rain???(N, 69, 73)
                torch.set_printoptions(profile="full")

                time = self.generateLabelOneHot(time, shape=(time.shape[0], 8))
                y_hat, time_hat = self.net(input)  #y_hat is (N, 4, 69, 73)
                #???????????????
                #TODO: ???????????????????????????????????????
                y = torch.zeros_like(y_hat).to('cuda')
                for k in range(self.threshold.shape[0]):
                    # if rain[j] > self.threshold[k]:
                    #     y[j][k] = 1
                    y[:, k] = rain > self.threshold[k]
                with torch.no_grad():
                    mask = self.get_mask(rain)  #???rain???-99999???mask???
                    mask2 = self.generateResampleMask(
                        rain)  #???rain???0???95%????????????mask

                #time_hat = F.softmax(time_hat)
                #print(time, time_hat)
                optimizer.zero_grad()
                ce, _ = FocalLoss()(y_hat, y, mask * mask2)
                loss_time = nn.CrossEntropyLoss()(time_hat, time)

                loss = ce + 10 * loss_time  #?????????ce???emd????????????; ???????????????????????????mask???????????????
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
                #TODO: ??????rain???temp?????????-99999??????????????????mask??????-99999 :tick

            print('total_loss:{}'.format(np.mean(losses)))
            self.writer.add_scalar("epoch_Loss",
                                   np.mean(losses),
                                   global_step=step)
            #???100???epoch???save
            if step % 100 == 0 and step != 0:
                torch.save(self.net.state_dict(),
                           save_path[:-4] + '{}'.format(step) + '.pth')
                temp_evaluate_loss = self.unet_evaluate()
                if temp_evaluate_loss < evaluate_loss:
                    evaluate_loss = temp_evaluate_loss
                    torch.save(self.net.state_dict(), save_path)
                self.net.train()
        self.writer.flush()
        #torch.save(self.net.encoder.state_dict(), save_path1)
        #torch.save(self.net.decoder.state_dict(), save_path2)
        return

    def unet_evaluate(self):
        total_steps = 0
        losses = []
        self.net.eval()
        for i, iter in enumerate(tqdm(self.evaluate_iter)):
            with torch.no_grad():
                input, rain, temp, time = iter
                input = input.type(torch.FloatTensor).to(self.device)
                rain = rain.type(torch.FloatTensor).to(self.device)
                y_hat, time = self.net(input)
                mask = self.get_mask(rain)  #???rain???mask
                #TODO???????????????????????????????????????????????????
                #?????????????????????ordinalTrainer.py???237???
                # predict_rain_value = regression_value(y_hat)
                # loss = nn.MSELoss()(predict_rain_value * mask, input * mask)

                y = torch.zeros_like(y_hat).to(self.device)
                for k in range(self.threshold.shape[0]):
                    # if rain[j] > self.threshold[k]:
                    #     y[j][k] = 1
                    y[:, k] = rain > self.threshold[k]
                #loss????????????focalloss????????????
                loss, _ = FocalLoss()(y_hat, y, mask)
                #??????loss??????????????????mask????????????????????????
                loss = loss / torch.sum(mask)
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

    #??????????????????-99999??????0,?????????1???mask
    def get_mask(self, x):
        zero = torch.zeros_like(x)
        ones = torch.ones_like(x)

        x = torch.where(x > -99999, ones, x)
        x = torch.where(x == -99999, zero, x)
        return x

    # ???batchsize*69*73?????????????????????mask
    def generateResampleMask(self, rain):
        #able_list = [0, 0, 0, 0, 0]
        zeros = torch.zeros_like(rain)
        ones = torch.ones_like(rain)
        result = ones
        if torch.rand(1) > 0.05:
            result = torch.where(rain == 0, zeros, ones)
        #print(result)
        return result

    #?????????????????????one-hot
    def generateLabelOneHot(self, x, shape=(32, 8)):
        result = torch.zeros(shape).to(self.device)
        for idx, item in enumerate(x):
            result[idx][int(item)] = 1
        #print(x, result)
        return result.float()

    def generateOneHot(self, softmax):
        maxIdxs = torch.argmax(softmax, dim=1, keepdim=True).cpu().long()
        oneHotMask = torch.zeros(softmax.shape, dtype=torch.float32)
        oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
        #oneHotMask = oneHotMask.unsqueeze(-2)
        return oneHotMask
