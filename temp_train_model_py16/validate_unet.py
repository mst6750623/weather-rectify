import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import yaml
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
#from net.conv_unet_original import ConvUNet
from net.conv_unet_spatial import ConvUNet
from dataset import gridDataset


#注意，这是validate，不是最终生成的test
class Validate(nn.Module):
    def __init__(self, args, data_iter, device):
        super(Validate, self).__init__()
        self.unet = ConvUNet(args['in_channels'], args['n_classes'])
        self.data_iter = data_iter
        self.device = device
        self.n_classes = args['n_classes']

    def initialize(self, unet_path):
        unet_ckpt = torch.load(unet_path)
        print('loading checkpoint:', unet_path, unet_ckpt)

        self.unet.load_state_dict(unet_ckpt)
        self.unet.eval()

    def simple_validate(self):
        total_points = 0
        valid_points = 0
        loss = 0
        loss_time = 0
        count = 0
        hit = 0
        for i, iter in enumerate(tqdm(self.data_iter, desc="validating: ")):
            with torch.no_grad():
                input, _, temp, time = iter
                input = input.type(torch.FloatTensor).to(self.device)
                temp = temp.type(torch.FloatTensor).to(self.device)
                #time = self.generateLabelOneHot(time, shape=(time.shape[0], 8))
                pred_temperature = self.unet(input)
                mask = self.get_mask(temp)
                loss += nn.L1Loss(reduction='sum')(mask * temp,
                                                   mask * pred_temperature)

                # pred_time = torch.argmax(pred_time, dim=1)
                # for j in range(time.shape[0]):
                #     if time[j][pred_time[j]] == 1:
                #         hit += 1

                total_points += temp.shape[0] * temp.shape[1] * temp.shape[2]
                valid_points += torch.sum(mask)

                #count += time.shape[0]

        print('Valid total points: {} - {} = {}'.format(
            total_points, total_points - valid_points, valid_points))
        print('Classification loss per point: ',
              loss / valid_points)  #正是网站的评分结果
        # print('Time prediction accuracy: {} / {} = {}'.format(
        #     hit, count, hit / count))

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


if __name__ == '__main__':
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    evaluate_dataset = gridDataset(config['train_dir'],
                                   isTrain=False,
                                   isFirstTime=False,
                                   nwp_num=config['unet']['in_channels'])

    evaluate_iter = DataLoader(evaluate_dataset,
                               batch_size=256,
                               shuffle=True,
                               pin_memory=True)
    device = 'cuda'
    validate = Validate(config['unet'], evaluate_iter, device).to(device)
    #validate.initialize('checkpoint/unet_lr0405_801.pth')
    validate.initialize('checkpoint/unet_0223_58_spatial_test_best.pth')
    #validate.forward()
    validate.simple_validate()
