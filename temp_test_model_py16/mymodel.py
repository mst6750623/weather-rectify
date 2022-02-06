import torch
import os
import yaml
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from xarray import open_dataset
from net.conv_unet import ConvUNet


class model(nn.Module):
    def __init__(self, device='cuda'):
        super(model, self).__init__()
        self.needed = [0, 8, 14, 17, 22, 28, 31, 35, 40]
        self.config = yaml.load(open('config.yaml', 'r'),
                                Loader=yaml.FullLoader)
        args = self.config['unet']
        self.mean = torch.load('processed_data/mean.pth').numpy()
        self.std = torch.load('processed_data/std.pth').numpy()

        self.device = device

        self.net = ConvUNet(args['in_channels'], args['n_classes']).to(device)
        self.initialize('checkpoint/unet_lr0405_05816.pth')

    def initialize(self, unet_path):
        unet_ckpt = torch.load(unet_path)

        self.net.load_state_dict(unet_ckpt)
        print('loading checkpoint:', unet_path)
        self.net.eval()

    def forward(self, input_dir):
        file_dir_name = input_dir
        result_list = []
        for j in range(9):
            loc_file_name = os.path.join(
                file_dir_name,
                'ji_loc_inputs_' + '{:0>2d}'.format(j + 1) + '.txt')
            input_file_name = os.path.join(
                file_dir_name,
                'grid_inputs_' + '{:0>2d}'.format(j + 1) + '.nc')
            if not os.path.exists(input_file_name) or not os.path.exists(
                    loc_file_name):
                continue
            temp_list = []

            with torch.no_grad():
                input_list = self.get_input_list(input_file_name)
                input_list = torch.from_numpy(input_list).to(self.device)
                input_list = input_list.unsqueeze(0)
                idx = 0
                prediction, _ = self.net(input_list)
                prediction = prediction.cpu().numpy()

            for line in open(loc_file_name):
                line = line.strip().split()
                row, col = line
                row, col = int(row), int(col)
                point_predicion = prediction[row][col]
                temp_list.append(point_predicion)
                #print(point_predicion.shape)
            result_list.append(np.asarray(temp_list))
        return np.asarray(result_list)

    def get_input_list(self, input_file_name):
        input = open_dataset(input_file_name)
        input_values = self.read_file(input)
        temp_list = []
        i = 0
        for values in input_values:
            if values.ndim == 3:
                values = np.transpose(values, (2, 0, 1))
                for num in range(values.shape[0]):
                    #temp_list.append(values[num].flatten().tolist())
                    if i in self.needed:
                        temp_list.append((values[num] - self.mean[i]) /
                                         self.std[i].tolist())
                    i += 1
            else:
                #temp_list.append(values.flatten().tolist())
                temp_list.append(
                    (values - self.mean[i]) / self.std[i].tolist())
                i += 1
        input = np.asarray(temp_list)
        return input

    def read_file(self, file):
        vars_name = [name for name in file]
        return [file.data_vars[name].values for name in vars_name]

    def generateOneHot(self, softmax):
        maxIdxs = torch.argmax(softmax, dim=1, keepdim=True).cpu().long()
        oneHotMask = torch.zeros(softmax.shape, dtype=torch.float32)
        oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
        #oneHotMask = oneHotMask.unsqueeze(-2)
        return oneHotMask
