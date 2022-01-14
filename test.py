import torch
import os
import yaml
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from xarray import open_dataset
from net.CombinatorialNetwork import CombinatorialNet
from net.confidence import confidenceNetwork
from newdataset import gridNewDataset


class Test():
    def __init__(self, combinatorial_args, device):
        self.needed = [0, 8, 14, 17, 22, 28, 31, 35, 40]
        self.mean = torch.load(
            '/mnt/pami23/stma/weather/processed_data/mean.pth').numpy()
        self.std = torch.load(
            '/mnt/pami23/stma/weather/processed_data/std.pth').numpy()
        self.confidence = confidenceNetwork().to(device)
        self.prediction = CombinatorialNet(
            combinatorial_args['encoder']['in_channels'],
            combinatorial_args['encoder']['mid_channels'],
            combinatorial_args['encoder']['out_channels'],
            combinatorial_args['ordinal']['mid_channels'],
            combinatorial_args['ordinal']['out_channels'],
            combinatorial_args['decoder']['mid_channels'],
            combinatorial_args['decoder']['out_channels'],
            combinatorial_args['nclass'],
            noise_mean=0,
            noise_std=1e-1).to(device)
        self.device = device
        self.nClass = combinatorial_args['nclass']

    def initialize(self, confidence_path, encoder_path, decoder_path,
                   ordinal_path):
        confidence_ckpt = torch.load(confidence_path)
        encoder_ckpt = torch.load(encoder_path)
        decoder_ckpt = torch.load(decoder_path)
        ordinal_ckpt = torch.load(ordinal_path)
        self.confidence.load_state_dict(confidence_ckpt)
        self.confidence.eval()
        self.prediction.encoder.load_state_dict(encoder_ckpt)
        self.prediction.decoder.load_state_dict(decoder_ckpt)
        self.prediction.OD.load_state_dict(ordinal_ckpt)
        self.prediction.eval()

    def test(self):
        config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
        test_path = config['test_dir']
        out_path = '/mnt/pami23/stma/weather/output/0114/'
        for i in tqdm(range(6, 400, 1)):
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
                out_file_name = os.path.join(
                    write_dir_name, 'pred_' + '{:0>2d}'.format(j + 1) + '.txt')
                with open(
                        out_file_name, 'w'
                ) as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！

                    for line in open(loc_file_name):
                        line = line.strip().split()
                        row, col = line
                        row, col = int(row), int(col)
                        #print(row, col)
                        temp_input = input_list[:, row - 8:row + 9,
                                                col - 8:col + 9].unsqueeze(0)
                        #print(temp_input.shape)
                        confidence_result = self.confidence(temp_input)
                        ordinal_results = self.prediction(temp_input,
                                                          isOrdinal=True)
                        #print(confidence_result, ordinal_results)
                        rainPredsSoftMax = F.softmax(confidence_result, dim=1)
                        rainOnehot = self.generateOneHot(rainPredsSoftMax).to(
                            self.device)
                        ordinal_results = ordinal_results[0]
                        if ordinal_results[0] < 0.5:
                            prediction = 0
                        elif ordinal_results[0] > 0.5 and ordinal_results[
                                1] < 0.5:
                            prediction = 0.1
                        elif ordinal_results[1] > 0.5 and ordinal_results[
                                2] < 0.5:
                            prediction = 3
                        elif ordinal_results[2] > 0.5 and ordinal_results[
                                3] < 0.5:
                            prediction = 10
                        else:
                            prediction = 20
                        f.write(
                            str(row) + ' ' + str(col) + ' ' + str(prediction) +
                            '\n')
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


if __name__ == "__main__":
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    device = 'cuda'
    test = Test(config['combinatotorial'], device)
    test.test()
