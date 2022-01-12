import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from net.CombinatorialNetwork import CombinatorialNet
from net.confidence import confidenceNetwork


#注意，这是validate，不是最终生成的test
class Validate(nn.Module):
    def __init__(self, combinatorial_args, data_iter, device):
        super(Validate, self).__init__()
        self.confidence = confidenceNetwork()
        self.prediction = CombinatorialNet(
            combinatorial_args['encoder']['in_channels'],
            combinatorial_args['encoder']['mid_channels'],
            combinatorial_args['encoder']['out_channels'],
            combinatorial_args['ordinal']['mid_channels'],
            combinatorial_args['ordinal']['out_channels'],
            combinatorial_args['decoder']['mid_channels'],
            combinatorial_args['decoder']['out_channels'],
            combinatorial_args['nclass'],
            noise_mean=0,
            noise_std=1e-1)
        self.data_iter = data_iter
        self.device = device
        self.nClass = combinatorial_args['nClass']

    def initialize(self, confidence_path, encoder_path, decoder_path,
                   ordinal_path):
        confidence_ckpt = torch.load(confidence_path)
        encoder_ckpt = torch.load(encoder_path)
        decoder_ckpt = torch.load(decoder_path)
        ordinal_ckpt = torch.load(ordinal_path)
        self.confidence.load_state_dict(confidence_ckpt)
        self.confidence.eval()
        self.prediction.encoder.load_state_dict(encoder_ckpt)
        self.prediction.decoder.load_state_dict(decoder_ckpt)
        self.prediction.OD.load_state_dict(ordinal_ckpt)
        self.prediction.eval()

    def forward(self):

        tsthreas = [0.1, 3, 10, 20]

        tp = [0] * len(tsthreas)  # true positive
        tn = [0] * len(tsthreas)  # true negetive
        fp = [0] * len(tsthreas)  # false positve
        fn = [0] * len(tsthreas)  # false negetive
        ts = [0] * len(tsthreas)
        totalRegressionLoss = []  #prediction loss
        totalReconstructLoss = []
        totalClassificationLoss = []  #传统regression loss
        totalRClassificationLoss = []  #confidence网络 loss
        total_error = 0
        total_count = 0
        p_error = 0
        ps_error = 0
        p_count = 0

        pxErrorList = [0] * (self.nClass)
        pxsErrorList = [0] * (self.nClass)
        pxCountList = [0] * (self.nClass)
        pxAverageError = [0] * (self.nClass)
        pxsAverageError = [0] * (self.nClass)

        classCorrect = [0] * (self.nClass)
        classCounnt = [0] * (self.nClass)
        accuray = [0] * (self.nClass + 1)

        rainCorrect = [0] * 2
        rainCount = [0] * 2
        rainAccuracy = [0] * 3

        for i, iter in enumerate(tqdm(self.data_iter), desc="validating: "):
            input, rain, temp = iter
            input = input.type(torch.FloatTensor).to(self.device)
            rain = rain.type(torch.FloatTensor).to(self.device)
            #TODO: 这里要对标签数据进行分类处理，得到rainClass
            rainClass = rain
            rainNumpy = rainClass.numpy()
            gt_micaps = rain.numpy
            with torch.no_grad():
                #三个网络分别的输出
                reconstructValues = self.prediction(input, isOrdinal=False)
                predictValues = self.prediction(input, isOrdinal=True)
                rainPreds = self.confidence(input)

                #得到confidence网络的预测oneHot，是否下雨，还有个普通regressionPredict的方法预测是否下雨，暂时没用
                rainPredsSoftMax = F.softmax(rainPreds, dim=1)
                rainOnehot = self.generateOneHot(rainPredsSoftMax).to(
                    self.device)

                #需要修改：看ordinal网络的输出
                regressionValues = 0.5 * (torch.sum(
                    (predictValues > 0.5).float(), dim=1).view(-1, 1))
                zeros = torch.zeros(regressionValues.size()).to(self.device)

                # 将confidence网络的预测进行mask，得到最终的输出
                regressionValues = torch.matmul(
                    rainOnehot,
                    torch.cat([zeros, regressionValues],
                              dim=1).unsqueeze(-1)).squeeze(-1)

                #regressionValues = regressionValues.item()
                #计算重建损失和预测与标签的损失
                regressionLoss = nn.MSELoss()(regressionValues, rain)
                reconstructLoss = nn.MSELoss()(reconstructValues, input)

                #TODO: 这里要用上分类处理后的标签数据
                #计算confidence损失
                rainClassificationLoss = nn.CrossEntropyLoss()(rainPreds,
                                                               rainClass)

                #更新三个损失的总体list
                totalRegressionLoss.append(regressionLoss.item())
                totalReconstructLoss.append(reconstructLoss.item())
                totalRClassificationLoss.append(rainClassificationLoss.item())
                '''
                #这是对普通regression做的
                for i in range(self.nClass):
                    classCorrect[i] += np.sum(
                        (predicted == i) * (regressionNumpy == i) *
                        (rainNumpy == 1))
                    classCounnt[i] += np.sum(
                        (regressionNumpy == i) * (rainNumpy == 1))
                '''
                #计算confidence预测准确的个数
                rainPredicted = torch.argmax(rainPredsSoftMax,
                                             dim=1).cpu().numpy()
                for i in range(2):
                    rainCorrect[i] += np.sum(
                        (rainPredicted == i) * (rainNumpy == i))
                    rainCount[i] += np.sum(rainNumpy == i)

                #计算最终输出与GT的预测差距
                predictNumpy = regressionValues.cpu().numpy()
                gapValues = np.abs(predictNumpy - gt_micaps)

                total_error += np.sum(gapValues)
                total_count += gapValues.shape[0]

                # 这里不知道输出的是啥,p_ae是一个gt中有降水（>0.05的区域）的点的值为预测差距值的array
                # p_error是有降水的点预测差的和
                p_ae = (gt_micaps > 0.05) * gapValues
                p_error += np.sum(p_ae)
                ps_error += np.sum(p_ae**2)
                p_count += np.sum(gt_micaps > 0.05)
                '''
                # 这是对传统regression做的
                one_hot_mask = regressionMask.numpy()
                for i in range(self.nClass):
                    ae = one_hot_mask[:, i].reshape(-1, 1) * gapValues
                    pxErrorList[i] += np.sum(ae)
                    pxsErrorList[i] += np.sum(ae**2)
                    pxCountList[i] += np.sum(one_hot_mask[:, i])'''

                for i, threas in enumerate(tsthreas):
                    tp[i] += np.sum(
                        (gt_micaps >= threas) * (predictNumpy >= threas))
                    tn[i] += np.sum(
                        (gt_micaps < threas) * (predictNumpy < threas))
                    fp[i] += np.sum(
                        (gt_micaps < threas) * (predictNumpy >= threas))
                    fn[i] += np.sum(
                        (gt_micaps >= threas) * (predictNumpy < threas))

        #计算TS,四舍五入保留5位小数
        for i, _ in enumerate(tsthreas):
            ts[i] += round(tp[i] / (tp[i] + fp[i] + fn[i]), 5)

        #计算总体平均误差——最终输出与GT预测的差/测试集大小
        #p平均误差
        #p平方后的平均误差
        totalAverageError = round(total_error / total_count, 5)
        pAverageError = round(p_error / p_count, 5)
        psAverageError = round(ps_error / p_count - pAverageError**2, 5)

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
