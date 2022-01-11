import torch
import os
import argparse
import numpy as np
import yaml
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from dataset import gridDataset
from confidenceTrainer import ConfidenceTrainer
from combinatorialTrainer import CombinatorialTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--correlation", type=str, default="correlation.npy")
parser.add_argument(
    "--test",
    type=bool,
    default=False,
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default='checkpoint/confidence.pth',
)
opts = parser.parse_args()


def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

    device = 'cuda'
    epoch = config['epoch']

    dataset = gridDataset(config['train_dir'], isTrain=not opts.test)
    if opts.test:
        data_iter = DataLoader(dataset, batch_size=1, shuffle=not opts.test)
        trainer = ConfidenceTrainer(config['confidence'], data_iter,
                                    device).to(device)
        trainer.initialize(opts.checkpoint_path)
        trainer.confidence_evaluate()
    else:
        data_iter = DataLoader(dataset,
                               batch_size=config['batch_size'],
                               num_workers=config['num_workers'],
                               shuffle=not opts.test)
        #trainer = ConfidenceTrainer(config['confidence'], data_iter,device).to(device)
        trainer = CombinatorialTrainer(config['combinatotorial'], data_iter,
                                       device).to(device)
        trainer.encoder_train()
        #trainer.confidence_train()


if __name__ == "__main__":
    main()