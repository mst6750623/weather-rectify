import torch
import os
import yaml
import argparse
import numpy as np
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
def main():
    config_file = args.config + '.yaml'
    config = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    train_dir = config['train_dir']
    print(train_dir)
    no_record_tensor = torch.full((69, 73), -99999.).to('cuda')
    result_list = []
    for i in tqdm(range(1962)):
        file_name = train_dir + 'example' + '{:0>5d}'.format(i + 1) + '/'
        for j in range(9):
            grid_file_name = file_name + 'grid_inputs_' + '{:0>2d}'.format(
                j + 1) + '.nc'
            ds = open_dataset(grid_file_name)
            vars_name = [name for name in ds]
            vars_values = [ds.data_vars[name].values for name in vars_name]
            #针对每个物理量
            temp_list=[]
            for values in vars_values:
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
                    values = np.transpose(values,(2,0,1))
                    for num in range(values.shape[0]):
                        temp_list.append(values[num].flatten().tolist())
                else:
                    temp_list.append(values.flatten().tolist())
                
                result_list.append(np.asarray(temp_list).transpose().tolist())
                print(np.asarray(result_list).shape)
            new_tensor = torch.Tensor(result_list).to('cuda')
            print(new_tensor.shape)
            return
    #print(len(result_list))


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


if __name__ == "__main__":
    main()
