import torch
import os
import argparse
import numpy as np
import yaml
import torch.utils.data as data
from dataset import gridDataset
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--correlation", type=str, default="correlation.npy")

opts = parser.parse_args()


def main():
    if os.path.exists(opts.correlation):
        correlation = np.load('data/01data.npy')
        print(correlation.shape)
        print("load file ok!")
    else:
        print("correlation file is not exist!")
    device = 'cuda'
    config = 'config'
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    trainer = Trainer(config['encoder'], config['decoder'],
                      config['confidence']).to(device)

    epoch = config['epoch']
    #dataloader = data.dataloader()
    for iter in range(config['epoch']):
        x = torch.randn((16, 58, 69, 73)).to(device)
        #x = x.permute(0, 3, 1, 2)
        y_hat, confidence, decoder = trainer(x)
        print(y_hat.shape, confidence.shape, decoder.shape)


if __name__ == "__main__":
    main()