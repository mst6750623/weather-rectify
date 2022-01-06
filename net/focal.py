import torch
import torch.nn as nn
"""-------  Rain Focal Loss -------"""


class FocalLoss(nn.Module):
    def __init__(self,
                 nClass,
                 alpha=0.75,
                 gamma=2,
                 balance_index=-1,
                 size_average=True):
        super(FocalLoss, self).__init__()
        # nClass = 2
        self.nClass = nClass
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.size_average = size_average

        if self.alpha is None:
            # [2×1]
            self.alpha = torch.ones(self.nClass, 1)

        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.nClass
            # [1×1]
            self.alpha = torch.FloatTensor(alpha).unsqueeze(-1)
            self.alpha = self.alpha / self.alpha.sum()

        elif isinstance(self.alpha, float):
            # [1,1]
            alpha = torch.ones(self.nClass, 1)
            # [0.25, 0.25]
            alpha = alpha * (1 - self.alpha)
            # [0.25, 0.25]
            alpha[balance_index] = self.alpha
            self.alpha = alpha

        else:
            raise NotImplementedError

    #
    def forward(self, input, target):
        # get dimensionality (N × 2)
        if input.dim() > 2:
            # N × 1 × 2
            input = input.view(input.size(0), input.size(1), -1)
        # N × 1
        target = target.view(-1, 1)
        epsilon = 1e-10
        alpha = self.alpha

        # assign CUDA:X to alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        # re-scale the elements in last dim lie in the range [0, 1] and sum to 1
        # softmax again
        logit = F.softmax(input, dim=-1)

        # binary 0/1 [N × 1]
        idx = target.cpu().long()

        # make zero for Tensor[size(N, 2)]
        one_hot_key = torch.FloatTensor(target.size(0), self.nClass).zero_()

        # insert 1 to the first dim of one_hot_key according to idx
        # one-hot binary (0,1)/(1,0) Tensor[size(N, 2)]
        one_hot_key = one_hot_key.scatter_(1, idx, 1)

        # same CUDA: x
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        # the possibility for specific classifier(only save GT-classifier) [N × 1]
        pt = (one_hot_key * logit).sum(1) + epsilon
        # log pt
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]

        # Cal Rain Focal Loss for binary classifier ( make gradient close to 0 ) [N × 1]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        # Mean/Sum
        if self.size_average:
            #  [1]
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss