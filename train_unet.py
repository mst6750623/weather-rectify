import torch
import os
import argparse
import numpy as np
import yaml
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from newdataset import gridNewDataset
from unetTrainer import UNetTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="unet")
parser.add_argument("--modelname", type=str, default="unet")
parser.add_argument(
    "--test",
    type=bool,
    default=False,
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default='checkpoint/unet.pth',
)
opts = parser.parse_args()


def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

    device = 'cuda'
    epoch = config['epoch']

    train_dataset = gridNewDataset(config['train_dir'],
                                   isTrain=True,
                                   isFirstTime=False)
    evaluate_dataset = gridNewDataset(config['train_dir'],
                                      isTrain=False,
                                      isFirstTime=False)
    train_iter = DataLoader(train_dataset,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers'],
                            shuffle=True,
                            pin_memory=True)
    evaluate_iter = DataLoader(evaluate_dataset,
                               batch_size=32,
                               shuffle=False,
                               pin_memory=True)

    if opts.test:
        if opts.model == 'unet':
            trainer = UNetTrainer(config['unet'],
                                           train_iter, evaluate_iter, device,
                                           opts.modelname).to(device)
        else:
            print('There is no correlated model!')
    else:
        if opts.model == 'unet':
            trainer = UNetTrainer(config['unet'],
                                           train_iter, evaluate_iter, device,
                                           opts.modelname).to(device)
            trainer.unet_train(epoch=2,
                                  lr=1e-2,
                                  save_path='checkpoint/unet.pth')
        else:
            print('There is no correlated model!')
        #trainer.confidence_train()


if __name__ == "__main__":
    main()