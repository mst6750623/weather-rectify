import torch
import os
import yaml
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from xarray import open_dataset
from net.conv_unet import ConvUNet
from dataset import gridDataset


class Test():
    def __init__(self, args, device):
        self.needed = [0, 8, 14, 17, 22, 28, 31, 35, 40]
        self.tsthreas = [0.1, 3, 10, 20]
        self.mean = torch.load(
            '/mnt/pami23/stma/weather/processed_data/mean.pth').numpy()
        self.std = torch.load(
            '/mnt/pami23/stma/weather/processed_data/std.pth').numpy()

        self.device = device

        self.net = ConvUNet(args['in_channels'], args['n_classes']).to(device)

    def initialize(self, unet_path):
        unet_ckpt = torch.load(unet_path)

        self.net.load_state_dict(unet_ckpt)
        self.net.eval()

    def test(self):
        config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
        test_path = config['test_dir']
        #记得改掉！
        out_path = '/mnt/pami23/stma/weather/output/0115/'
        for i in tqdm(range(400)):
            file_dir_name = os.path.join(test_path,
                                         'example' + '{:0>5d}'.format(i + 1))
            write_dir_name = os.path.join(out_path,
                                          'example' + '{:0>5d}'.format(i + 1))
            if not os.path.exists(write_dir_name):
                os.mkdir(write_dir_name)
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
                input_list = self.get_input_list(input_file_name)
                input_list = torch.from_numpy(input_list).to(self.device)

                idx = 0
                prediction = self.net(input_list).to(self.device)
                out_file_name = os.path.join(
                    write_dir_name, 'pred_' + '{:0>2d}'.format(j + 1) + '.txt')

                with open(
                        out_file_name, 'w'
                ) as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！

                    for line in open(loc_file_name):
                        line = line.strip().split()
                        row, col = line
                        row, col = int(row), int(col)

                        point_predicion = prediction[row][col]
                        print(point_predicion.shape)

                        if point_predicion[0] < 0.5:
                            prediction = 0
                        elif point_predicion[0] > 0.5 and point_predicion[
                                1] < 0.5:
                            prediction = 0.1
                        elif point_predicion[1] > 0.5 and point_predicion[
                                2] < 0.5:
                            prediction = 3
                        elif point_predicion[2] > 0.5 and point_predicion[
                                3] < 0.5:
                            prediction = 10
                        else:
                            prediction = 20
                        f.write(str(prediction) + '\n')
                    f.close()

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

    def updateFile(self):
        out_path = '/mnt/pami23/stma/weather/output/0114/'
        new_path = '/mnt/pami23/stma/weather/output/0114new/'
        for i in tqdm(range(400)):
            new_dir_name = os.path.join(new_path,
                                        'example' + '{:0>5d}'.format(i + 1))
            write_dir_name = os.path.join(out_path,
                                          'example' + '{:0>5d}'.format(i + 1))
            if not os.path.exists(new_dir_name):
                os.mkdir(new_dir_name)
            for j in range(9):

                out_file_name = os.path.join(
                    write_dir_name, 'pred_' + '{:0>2d}'.format(j + 1) + '.txt')

                if not os.path.exists(out_file_name):
                    continue

                new_file_name = os.path.join(
                    new_dir_name, 'pred_' + '{:0>2d}'.format(j + 1) + '.txt')
                with open(
                        new_file_name, 'w'
                ) as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！

                    for line in open(out_file_name):
                        line = line.strip().split()
                        row, col, pred = line

                        #print(row, col, pred)

                        f.write(str(pred) + '\n')
                    f.close()


if __name__ == "__main__":
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    device = 'cuda'
    test = Test(config['unet'], device)
    test.initialize('checkpoint/unet.pth')
    test.test()
