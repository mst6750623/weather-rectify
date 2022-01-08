import torch
import os
import argparse
import numpy as np

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


if __name__ == "__main__":
    main()