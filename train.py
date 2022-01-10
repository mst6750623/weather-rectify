import torch
import os
import argparse
import numpy as np
import yaml
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from dataset import gridDataset
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--correlation", type=str, default="correlation.npy")
parser.add_argument(
    "--test",
    type=bool,
    default=False,
)
opts = parser.parse_args()


def main():
    '''if os.path.exists(opts.correlation):
        correlation = np.load('data/01data.npy')
        print(correlation.shape)
        print("load file ok!")
    else:
        print("correlation file is not exist!")'''
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

    device = 'cuda'
    epoch = config['epoch']
    trainer = Trainer(config['encoder'], config['decoder'],
                      config['confidence']).to(device)
    dataset = gridDataset(config['train_dir'], isTrain=not opts.test)
    data_iter = DataLoader(dataset,
                           batch_size=config['batch_size'],
                           shuffle=not opts.test)
    for iter in tqdm(range(epoch)):
        for i, iter in enumerate(data_iter):
            if i == 10:
                return
            [input, rain, temp] = iter
            #原本是double的，但网络参数是float，不改输入的话，就得在网络参数上手动改（较为麻烦）
            input = input.type(torch.FloatTensor).to(device)
            rain = rain.type(torch.FloatTensor).to(device)
            temp = temp.type(torch.FloatTensor).to(device)
            y_hat, confidence, decoder = trainer(input)
            #print(y_hat.shape, confidence.shape, decoder.shape)


if __name__ == "__main__":
    main()