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
    def __init__(self, data_path, isTrain=True, isFirstTime=False, nwp_num=22):
        total_len = 0
        self.time_class = [0, 3, 6, 9, 12, 15, 18, 21]
        class_num = 8
        if isFirstTime:

            self.input = []
            self.rain = []
            self.temp = []
            self.time = []
            for i in tqdm(range(1960), desc="Scanning dataset files"):
                file_name = data_path + 'example' + '{:0>5d}'.format(i +
                                                                     1) + '/'
                #由于是预测12-36小时，所以起始时间+12
                start_time = 12
                if os.path.exists(os.path.join(file_name, '12_12-36h')):
                    start_time = 0
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
                        start_time += 3
                        continue
                    time_classification = torch.Tensor(
                        [self.time_class.index(start_time)]).long()
                    time_onehot = torch.zeros(class_num).scatter_(
                        0, time_classification, 1)
                    self.input.append(input_file_name)
                    self.rain.append(rain_file_name)
                    self.temp.append(temp_file_name)
                    self.time.append(time_onehot)
                    total_len += 1
                    start_time += 3
                    if start_time >= 24:
                        start_time -= 24
            np.save('processed_data/input.npy', self.input)
            np.save('processed_data/rain.npy', self.rain)
            np.save('processed_data/temp.npy', self.temp)
            np.save('processed_data/time.npy', self.time)
            #计算所有输入数据的mean和std进行数据归一化
            mean = torch.zeros(58)
            std = torch.zeros(58)
            for idx in tqdm(range(len(self.input)),
                            desc='Calculating mean and std'):
                input = open_dataset(self.input[idx])
                input_values = read_file(input)
                i = 0
                for values in input_values:
                    if values.ndim == 3:
                        values = np.transpose(values, (2, 0, 1))
                        for num in range(values.shape[0]):
                            #temp_list.append(values[num].flatten().tolist())
                            #temp_list.append(values[num].tolist())

                            mean[i] += values[num].mean()
                            std[i] += values[num].std()
                            i += 1
                    else:
                        #temp_list.append(values.flatten().tolist())
                        #temp_list.append(values.tolist())
                        mean[i] += values.mean()
                        std[i] += values.std()
                        i += 1
            mean.div_(len(self.input))
            std.div_(len(self.input))
            print(mean, std)
            torch.save(mean, 'processed_data/mean.pth')
            torch.save(std, 'processed_data/std.pth')
        else:
            self.input = np.load('processed_data/input.npy')
            self.rain = np.load('processed_data/rain.npy')
            self.temp = np.load('processed_data/temp.npy')
            self.time = np.load('processed_data/time.npy', allow_pickle=True)
            total_len = self.input.shape[0]
            self.mean = torch.load('processed_data/mean.pth').numpy()
            self.std = torch.load('processed_data/std.pth').numpy()

        train_len = int(0.9 * total_len)
        if isTrain:
            self.input = self.input[:train_len]
            self.rain = self.rain[:train_len]
            self.temp = self.temp[:train_len]
            self.time = self.time[:train_len]
            self.length = int(0.9 * total_len)
        else:
            self.input = self.input[train_len:]
            self.rain = self.rain[train_len:]
            self.temp = self.temp[train_len:]
            self.time = self.time[train_len:]
            self.length = int(0.1 * total_len)

        self.nwp_num = nwp_num
        if nwp_num >= 58:
            self.needed = range(45)
        elif nwp_num == 22:
            self.needed = [0, 8, 14, 17, 22, 28, 31, 35, 40]
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

        if self.nwp_num == 59:
            temp_list.append(rain_values[0].tolist())
        input = np.asarray(temp_list)

        input = torch.from_numpy(input)
        rain = torch.from_numpy(rain_values[0])
        temp = torch.from_numpy(temp_values[0])
        time = self.time[idx]
        return input, rain, temp, time


if __name__ == "__main__":
    dataset = gridDataset("/mnt/pami23/stma/weather/train/",
                          isTrain=False,
                          isFirstTime=False,
                          nwp_num=58)
    # time_class = [0, 3, 6, 9, 12, 15, 18, 21]
    # a = 12
    # class_num = 8
    # time_classification = torch.Tensor([time_class.index(a)]).long()

    # print(time_classification)
    # one_hot = torch.zeros(class_num).scatter_(0, time_classification, 1)
    # print(one_hot)
    '''mean = torch.zeros(58)
    std = torch.zeros(58)
    for idx in tqdm(range(dataset.length)):
        input = open_dataset(dataset.input[idx])

        input_values = read_file(input)
        i = 0
        for values in input_values:
            if values.ndim == 3:
                values = np.transpose(values, (2, 0, 1))
                for num in range(values.shape[0]):
                    #temp_list.append(values[num].flatten().tolist())
                    #temp_list.append(values[num].tolist())

                    mean[i] += values[num].mean()
                    std[i] += values[num].std()
                    i += 1
            else:
                #temp_list.append(values.flatten().tolist())
                #temp_list.append(values.tolist())
                mean[i] += values.mean()
                std[i] += values.std()
                i += 1

    mean.div_(dataset.length)
    std.div_(dataset.length)
    print(mean, std)
    torch.save(mean, 'processed_data/mean.pth')
    torch.save(std, 'processed_data/std.pth')'''
    input, rain, temp, time = dataset.__getitem__(784)
    print(input.shape, rain.shape, temp.shape, time)
