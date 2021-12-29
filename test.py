import torch
import os
import yaml
from tqdm import tqdm
from xarray import open_dataset



def main():
    config = 'config'
    config = yaml.load(open('config.yaml', 'r'),Loader=yaml.FullLoader)
    train_dir = config['train_dir']
    print(train_dir)
    no_record = torch.full((69,73),-99999.).to('cuda')
    result_list = []
    for i in tqdm(range(1962)):
        file_name = train_dir+'example'+'{:0>5d}'.format(i+1)+'/'
        #if i%100 ==0:
        #    print('range: ',i)
        for j in range(9):
            temp_file_name = file_name+'obs_grid_rain'+'{:0>2d}'.format(j+1)+'.nc'
            ds = open_dataset(temp_file_name)
            vars_name = [name for name in ds]
            vars_values = [ds.data_vars[name].values for name in vars_name]
            for values in vars_values:
                new_tensor = torch.from_numpy(values).to('cuda')
                #print(new_tensor)
                if new_tensor.equal(no_record):
                    print('i: ',i,' j: ',j)
    #print(len(result_list))

def print_single_point():
    file_ = "/mnt/pami23/stma/weather/train/example01002/grid_inputs_05.nc"
    ds = open_dataset(file_)
    vars_name = [name for name in ds]

    print(len(vars_name))

    #获 得 变 量 值 列 表, 每 个 变 量 值 是numpy多 维 数 组.
    #真 实 值 格 点 资 料 中 的-99999.0代 表 缺 测 值.
    vars_values = [ds.data_vars[name].values for name in vars_name]
    for i,values in enumerate(vars_values):
        new_tensor = torch.from_numpy(values)
        print(new_tensor[20][20])

if __name__ == "__main__":
    print_single_point()
        
    
        
    