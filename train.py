import torch
import os
import argparse
import numpy as np
import yaml
from trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--correlation", type=str, default="correlation.npy")

opts = parser.parse_args()


def main():
    if os.path.exists(opts.correlation):
        correlation = np.load(opts.correlation)
        print(correlation.shape)
        print("load file ok!")
    else:
        print("correlation file is not exist!")

    config = 'config'
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    trainer = Trainer(config['encoder'], config['decoder'],
                      config['confidence'])
    trainer.forward()


if __name__ == "__main__":
    main()