import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import yaml
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from net.conv_unet import ConvUNet
from dataset import gridDataset


#注意，这是validate，不是最终生成的test
class Validate(nn.Module):
    def __init__(self, args, data_iter, device):
        super(Validate, self).__init__()
        self.unet = ConvUNet(args['in_channels'], args['n_classes'])
        self.data_iter = data_iter
        self.device = device
        self.n_classes = args['n_classes']

    def initialize(self, unet_path):
        unet_ckpt = torch.load(unet_path)

        self.unet.load_state_dict(unet_ckpt)
        self.unet.eval()

    def simple_validate(self):

        tsthreas = torch.tensor([0.1, 3, 10, 20]).to(self.device)

        tp = [0] * len(tsthreas)  # true positive
        tn = [0] * len(tsthreas)  # true negetive
        fp = [0] * len(tsthreas)  # false positve
        fn = [0] * len(tsthreas)  # false negetive
        ts = [0] * len(tsthreas)


        for i, iter in enumerate(tqdm(self.data_iter, desc="validating: ")):
            input, rain, _ = iter
            input = input.type(torch.FloatTensor).to(self.device)
            rain = rain.type(torch.FloatTensor).to(self.device)  #(N, 1)

            #只算一下TS
            with torch.no_grad():
                pred_classification_scores = self.unet(input) #scores: (N, 4, H, W)
                regression_value = None #TODO: 把分类转为具体数值
                mask = self.get_mask(rain)
                print(torch.sum(mask))
                threshold_for_probability = 0.7

                for j, threas in enumerate(tsthreas):
                    tp[j] += torch.sum(
                        (mask * (rain >= threas)) * (pred_classification_scores[:, j] >=
                                                 threshold_for_probability))
                    tn[j] += torch.sum(
                        (mask * (rain < threas)) * (pred_classification_scores[:, j] <
                                                threshold_for_probability))
                    fp[j] += torch.sum(
                        (mask * (rain < threas)) * (pred_classification_scores[:, j] >=
                                                threshold_for_probability))
                    fn[j] += torch.sum(
                        (mask * (rain >= threas)) * (pred_classification_scores[:, j] <
                                                 threshold_for_probability))
                #print('finals:', tp, tn, fp, fn)
        #计算TS（对于整个epoch而言）,四舍五入保留5位小数
        for j, threas in enumerate(tsthreas):
            ts[j] = tp[j] / (tp[j] + fp[j] + fn[j])
            print('For x = {}, ts = {}'.format(threas.cpu().numpy(), ts[j]))


    #生成将所有的-99999变成0,其他为1的mask
    def get_mask(self, x):
        zero = torch.zeros_like(x)
        ones = torch.ones_like(x)

        x = torch.where(x > -99999, ones, x)
        x = torch.where(x == -99999, zero, x)
        return x

    def generateOneHot(self, softmax):
        maxIdxs = torch.argmax(softmax, dim=1, keepdim=True).cpu().long()
        oneHotMask = torch.zeros(softmax.shape, dtype=torch.float32)
        oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
        #oneHotMask = oneHotMask.unsqueeze(-2)
        return oneHotMask


if __name__ == '__main__':
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    evaluate_dataset = gridDataset(config['train_dir'],
                                      isTrain=False,
                                      isFirstTime=False)

    evaluate_iter = DataLoader(evaluate_dataset,
                               batch_size=256,
                               shuffle=True,
                               pin_memory=True)
    device = 'cuda'
    validate = Validate(config['unet'], evaluate_iter,
                        device).to(device)
    validate.initialize('checkpoint_new/unet_lr05100.pth')
    #validate.forward()
    validate.simple_validate()
