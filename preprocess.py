import torch
import os
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from xarray import open_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config')
args = parser.parse_args()
'''
1229 todo:
1.计算物理量之间的corrcoef(),先遍历拼成一个numpy array保存到一个文件（用例*物理量）
2.对海拔不同的筛选出相关系数最高的
'''


def get_correlation(label_file):
    lbl = np.load(label_file)
    corr_ma = np.corrcoef(lbl.transpose())
    return corr_ma
    new_tensor = torch.from_numpy(corr_ma)
    print(new_tensor.shape, new_tensor)


def get_all_correlation(label):
    correlation = []
    for i in tqdm(range(40)):
        filename = 'data/' + '{:02d}'.format(i + 1) + 'data.npy'
        if i == 0:
            correlation = get_correlation(filename)
        else:
            correlation += get_correlation(filename)
    correlation = correlation / 40
    print(correlation)
    np.save('correlation.npy', correlation)


def main():
    config_file = args.config + '.yaml'
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    train_dir = config['train_dir']
    print(train_dir)
    #no_record_tensor = torch.full((69, 73), -99999.).to('cuda')
    result_list = []
    for i in tqdm(range(1962)):
        file_name = train_dir + 'example' + '{:0>5d}'.format(i + 1) + '/'
        for j in range(9):
            input_file_name = file_name + 'grid_inputs_' + '{:0>2d}'.format(
                j + 1) + '.nc'
            rain_file_name = file_name + 'obs_grid_rain' + '{:0>2d}'.format(
                j + 1) + '.nc'
            temp_file_name = file_name + 'obs_grid_temp' + '{:0>2d}'.format(
                j + 1) + '.nc'
            #如果某个文件不存在，就跳过不要这个数据了吧
            if not os.path.isfile(input_file_name) or not os.path.isfile(
                    rain_file_name) or not os.path.isfile(temp_file_name):
                continue

            input = open_dataset(input_file_name)
            rain = open_dataset(rain_file_name)
            temp = open_dataset(rain_file_name)

            input_values = read_file(input)
            rain_values = read_file(rain)
            temp_values = read_file(temp)
            #针对每个物理量
            temp_list = []
            for values in input_values:
                '''new_tensor = torch.from_numpy(values).to('cuda')
                if new_tensor.dim()==3:
                    new_tensor = new_tensor.permute(2,0,1)
                    #针对有多个维度的物理量
                    for num in range(new_tensor.shape[0]):
                        temp_list.append(new_tensor[num].reshape(1,-1).squeeze().tolist())
                else:
                    temp_list.append(new_tensor.reshape(1,-1).squeeze().tolist())

                #print(torch.from_numpy(temp_list.transpose()).shape)
                result_list.append(temp_list.)
                print(len(result_list))'''
                if values.ndim == 3:
                    values = np.transpose(values, (2, 0, 1))
                    for num in range(values.shape[0]):
                        #temp_list.append(values[num].flatten().tolist())
                        temp_list.append(values[num].flatten().tolist())
                else:
                    #temp_list.append(values.flatten().tolist())
                    temp_list.append(values.flatten().tolist())
            temp_list.append(rain_values[0].flatten().tolist())
            temp_list.append(temp_values[0].flatten().tolist())
            result_list += np.asarray(temp_list).transpose().tolist()
        if i % 40 == 0 and i != 0:
            np_result = np.asarray(result_list)
            name = '{:0>2d}'.format(int(i / 50)) + 'data'
            name = os.path.join('data_with_label', name)
            np.save(name + '.npy', np_result)
            result_list = []
            print(np_result.shape)
        #print(np.asarray(result_list).shape)

    #get_correlation('400data.npy')
    #print(len(result_list))
    np_result = np.asarray(result_list)
    name = '50data'
    name = os.path.join('data', name)
    np.save(name + '.npy', np_result)
    result_list = []
    print(np_result.shape)


# correlated heat map
def heat_map_plot(corrMatrix):
    # Create a figure and a set of subplots, Number of rows/columns of the subplot grid
    # fig → figure ax → axe can be either a single Axes object or an array of Axes objects if more than one subplot was created
    f, ax = plt.subplots(figsize=corrMatrix.shape)
    # cmap is a color palette
    cmap = sns.diverging_palette(255, 0, as_cmap=True)
    # use the column names but plot only every n label: e.g. temperature||wind||...
    # Axes in which to draw the plot, otherwise use the currently-active Axes
    sns.heatmap(corrMatrix, cmap=cmap, ax=ax)

    f.savefig("corrMatrix1.jpg")


def print_single_point():
    file_ = "/mnt/pami23/stma/weather/train/example01002/grid_inputs_05.nc"
    ds = open_dataset(file_)
    vars_name = [name for name in ds]

    print(len(vars_name))

    #获 得 变 量 值 列 表, 每 个 变 量 值 是numpy多 维 数 组.
    #真 实 值 格 点 资 料 中 的-99999.0代 表 缺 测 值.
    vars_values = [ds.data_vars[name].values for name in vars_name]
    for i, values in enumerate(vars_values):
        new_tensor = torch.from_numpy(values)
        print(new_tensor[20][20])


def read_file(file):
    vars_name = [name for name in file]
    return [file.data_vars[name].values for name in vars_name]


if __name__ == "__main__":
    #main()
    #get_all_correlation('')
    lbl_all = np.load('correlation.npy')
    #heat_map_plot(lbl_all)
    torch.set_printoptions(profile="full")
    lbl_sum = np.sum(lbl_all, axis=1)
    print(lbl_sum[0:45])
