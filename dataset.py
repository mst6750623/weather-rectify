import torch
import torch.utils.data as data
import numpy as np
import os
from tqdm import tqdm
from xarray import open_dataset


def read_file(file):
    vars_name = [name for name in file]
    return [file.data_vars[name].values for name in vars_name]


class gridDataset(data.Dataset):
    def __init__(self, data_path, isTrain=True, isFirstTime=False):
        total_len = 0
        if isFirstTime:
            # 在这边只扫一遍文件名
            self.input = []
            self.rain = []
            self.temp = []
            for i in tqdm(range(1962), desc="Scanning dataset files"):
                file_name = data_path + 'example' + '{:0>5d}'.format(i +
                                                                     1) + '/'
                for j in range(9):
                    input_file_name = file_name + 'grid_inputs_' + '{:0>2d}'.format(
                        j + 1) + '.nc'
                    rain_file_name = file_name + 'obs_grid_rain' + '{:0>2d}'.format(
                        j + 1) + '.nc'
                    temp_file_name = file_name + 'obs_grid_temp' + '{:0>2d}'.format(
                        j + 1) + '.nc'
                    #如果某个文件不存在，就跳过不要这个数据了吧
                    if not os.path.isfile(
                            input_file_name) or not os.path.isfile(
                                rain_file_name) or not os.path.isfile(
                                    temp_file_name):
                        continue

                    self.input.append(input_file_name)
                    self.rain.append(rain_file_name)
                    self.temp.append(temp_file_name)
                    total_len += 1
            np.save('input.npy', self.input)
            np.save('rain.npy', self.rain)
            np.save('temp.npy', self.temp)
        else:
            self.input = np.load('input.npy')
            self.rain = np.load('rain.npy')
            self.temp = np.load('temp.npy')
            total_len = self.input.shape[0]

        train_len = int(0.9 * total_len)
        if isTrain:
            self.input = self.input[:train_len]
            self.rain = self.rain[:train_len]
            self.temp = self.temp[:train_len]
            self.length = int(0.9 * total_len)
        else:
            self.input = self.input[train_len:]
            self.rain = self.rain[train_len:]
            self.temp = self.temp[train_len:]
            self.length = int(0.1 * total_len)

        print("length:", self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #在这边读路径下的数据

        input_file_name = self.input[idx]
        rain_file_name = self.rain[idx]
        temp_file_name = self.temp[idx]
        input = open_dataset(input_file_name)
        rain = open_dataset(rain_file_name)
        temp = open_dataset(temp_file_name)

        input_values = read_file(input)
        rain_values = read_file(rain)
        temp_values = read_file(temp)

        mean = torch.zeros(58)
        std = torch.zeros(58)
        temp_list = []
        i = 0
        for values in input_values:

            if values.ndim == 3:
                values = np.transpose(values, (2, 0, 1))
                for num in range(values.shape[0]):
                    #temp_list.append(values[num].flatten().tolist())
                    temp_list.append(values[num].tolist())
                    mean[i] += values[num].mean()
                    std[i] += values[num].std()
            else:
                #temp_list.append(values.flatten().tolist())
                temp_list.append(values.tolist())
                mean[i] += values.mean()
                std[i] += values.std()
            i += 1
        input = np.asarray(temp_list)

        mean.div_(self.length)
        std.div_(self.length)
        print(mean, std)

        input = torch.from_numpy(input)
        rain = torch.from_numpy(rain_values[0])
        temp = torch.from_numpy(temp_values[0])
        return input, rain, temp


if __name__ == "__main__":
    dataset = gridDataset("/mnt/pami23/stma/weather/train/", True, False)

    input, rain, temp = dataset.__getitem__(0)
    print(input.shape, rain.shape, temp.shape)
