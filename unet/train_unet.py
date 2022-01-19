import argparse
import yaml
from torch.utils.data.dataloader import DataLoader
from datasetwithtime import gridDataset
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

    train_dataset = gridDataset(config['train_dir'],
                                isTrain=True,
                                isFirstTime=False)
    evaluate_dataset = gridDataset(config['train_dir'],
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
            trainer = UNetTrainer(config['unet'], train_iter, evaluate_iter,
                                  device, opts.modelname).to(device)
        else:
            print('There is no correlated model!')
    else:
        if opts.model == 'unet':
            trainer = UNetTrainer(config['unet'], train_iter, evaluate_iter,
                                  device, opts.modelname).to(device)
            trainer.unet_train(epoch=config['epoch'],
                               lr=1e-5,
                               save_path='../checkpoint/unetwithtime.pth')
        else:
            print('There is no correlated model!')
        #trainer.confidence_train()


if __name__ == "__main__":
    main()