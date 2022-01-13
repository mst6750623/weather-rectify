import torch
import torch.utils.data as data
import numpy as np
import os
from tqdm import tqdm
from xarray import open_dataset


def read_file(file):
    vars_name = [name for name in file]
    return [file.data_vars[name].values for name in vars_name]


#TODO:
#改成对范围内每个站点的dataset :tick
#增加时间监督信息
# start_time分成8类[0,3,6,9,12,15,18,21]
class gridNewDataset(data.Dataset):
    def __init__(self, data_path, isTrain=True, isFirstTime=False):
        time_class = [0, 3, 6, 9, 12, 15, 18, 21]
        self.mean = torch.load(
            '/mnt/pami23/stma/weather/processed_data/mean.pth').numpy()
        self.std = torch.load(
            '/mnt/pami23/stma/weather/processed_data/std.pth').numpy()
        self.needed = [0, 8, 14, 17, 22, 28, 31, 35, 40]
        if isFirstTime:
            # 在这边只扫一遍文件名
            self.inputfile = []
            self.rainfile = []
            self.tempfile = []
            for i in tqdm(range(1962), desc="Scanning dataset files"):
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
                    time_classification = time_class.index(start_time)
                    self.inputfile.append([input_file_name, 0])
                    self.rainfile.append([rain_file_name, time_classification])
                    self.tempfile.append([temp_file_name, time_classification])
                    start_time += 3
                    if start_time >= 24:
                        start_time -= 24
            np.save('/mnt/pami23/stma/weather/processed_data/newInputFile.npy',
                    self.inputfile)
            np.save('/mnt/pami23/stma/weather/processed_data/newRainFile.npy',
                    self.rainfile)
            np.save('/mnt/pami23/stma/weather/processed_data/newTempFile.npy',
                    self.tempfile)
        else:
            self.inputfile = np.load(
                '/mnt/pami23/stma/weather/processed_data/newInputFile.npy')
            self.rainfile = np.load(
                '/mnt/pami23/stma/weather/processed_data/newRainFile.npy')
            self.tempfile = np.load(
                '/mnt/pami23/stma/weather/processed_data/newTempFile.npy')
        self.length = 0
        self.input = []
        self.rain = []
        self.temp = []
        self.time = []
        print("file len:", len(self.inputfile), len(self.rainfile),
              len(self.tempfile))
        if isFirstTime:
            for idx, [filename, _] in enumerate(tqdm(self.inputfile)):
                input_list = self.get_input_list(filename)
                rain_list = self.get_label_file(self.rainfile[idx][0])
                temp_list = self.get_label_file(self.tempfile[idx][0])
                time_classification = self.rainfile[idx][1]
                self.input.append(input_list)
                self.rain.append(rain_list)
                self.temp.append(temp_list)
                self.time.append(time_classification)
                '''for i in range(8, 61, 1):
                    for j in range(8, 65, 1):
                        self.input.append(input_list[:, i - 8:i + 8,
                                                     j - 8:j + 8])
                        self.rain.append(rain_list[i, j])
                        self.temp.append(temp_list[i, j])
                        self.time.append(time_classification)
                        self.length += 1'''
            #由于数据量过于大了，我们只存17646个总体文件
            np.save('/mnt/pami23/stma/weather/processed_data/newInput.npy',
                    self.input)
            np.save('/mnt/pami23/stma/weather/processed_data/newRain.npy',
                    self.rain)
            np.save('/mnt/pami23/stma/weather/processed_data/newTemp.npy',
                    self.temp)
            np.save('/mnt/pami23/stma/weather/processed_data/newTime.npy',
                    self.time)
        else:
            #self.input = np.load('processed_data/newInput.npy')

            print('loading input!')
            self.input = np.load(
                '/mnt/pami23/stma/weather/processed_data/newInput.npy')
            print('loading rain!')
            self.rain = np.load(
                '/mnt/pami23/stma/weather/processed_data/newRain.npy')
            print('loading temp!')
            self.temp = np.load(
                '/mnt/pami23/stma/weather/processed_data/newTemp.npy')
            print('loading time!')
            self.time = np.load(
                '/mnt/pami23/stma/weather/processed_data/newTime.npy')
        total_len = self.input.shape[0] * 53 * 57
        train_len = int(0.9 * total_len)
        file_len = len(self.inputfile)
        train_file_len = int(0.9 * file_len)
        if isTrain:
            self.input = self.input[:train_file_len]
            self.rain = self.rain[:train_file_len]
            self.temp = self.temp[:train_file_len]
            self.time = self.time[:train_file_len]
            self.length = train_len
        else:
            self.input = self.input[train_file_len:]
            self.rain = self.rain[train_file_len:]
            self.temp = self.temp[train_file_len:]
            self.time = self.time[train_file_len:]
            self.length = total_len - train_len
        print('length:', self.length)
        print(len(self.input))

    def get_input_list(self, input_file_name):
        input = open_dataset(input_file_name)
        input_values = read_file(input)
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

    def get_label_file(self, label_file_name):
        label = open_dataset(label_file_name)
        label_values = read_file(label)
        return label_values[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #53*57=3021
        real_id = int(idx / 3021)
        inner_id = int(idx % 3021)
        total_input = torch.from_numpy(self.input[real_id])
        total_rain = torch.from_numpy(self.rain[real_id])
        total_temp = torch.from_numpy(self.temp[real_id])
        time = self.time[real_id]

        i = int(inner_id / 57) + 8
        j = int(inner_id % 57) + 8
        input = total_input[:, i - 8:i + 9, j - 8:j + 9]
        rain = total_rain[i, j]
        temp = total_temp[i, j]

        return input, rain, temp, time


if __name__ == "__main__":
    dataset = gridNewDataset("/mnt/pami23/stma/weather/train/",
                             isTrain=True,
                             isFirstTime=False)
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
    input, rain, temp, time = dataset.__getitem__(523465)
    print(input.shape, rain, temp, time)
