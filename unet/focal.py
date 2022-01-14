import torch
import torch.nn as nn
from torch.nn import functional as F
"""-------  Ordinal Focal Loss -------"""


class FocalLoss(nn.Module):

    def __init__(self):
        super(FocalLoss, self).__init__()
        """------- term added to improve numerical stability -------"""
        self.epilson = 1e-10
        self.gamma = 2
        #self.space = space

    def forward(self, x, y):
        batchSize = x.size(0)

        gamma = self.gamma
        ##  Mean Focal Loss with CE
        ce = -(torch.sum(
            torch.pow((1 - x), gamma) * torch.log(x + self.epilson).mul(y) +
            torch.pow(x, gamma) *
            torch.log(1 - x + self.epilson).mul(1 - y))) / batchSize

        # cal histogram distribution for all x
        x_soft = F.softmax(x, dim=1)
        # cal histogram distribution for all y
        y_soft = F.softmax(y, dim=1)

        ## EMD (Earth Mover's Distance) Loss
        x_cumsum = torch.cumsum(x_soft, dim=1)
        y_cumsum = torch.cumsum(y_soft, dim=1)

        EMD_loss = torch.sum((x_cumsum - y_cumsum)**2)
        return ce, EMD_loss


if __name__ == '__main__':
    input = torch.rand((8, 4, 69, 73))
    target = torch.rand((8, 4, 69, 73))
    OD = FocallLoss()
    print(OD(input, target))
