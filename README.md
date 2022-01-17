# weather-rectify
Code for weather rectify competition


## TODO: 
- [x] 训练三个网络
- [x] ordinal网络trainer构建
- [x] validate模块构建
- [x] 跑测试集！
- [x] 重新构建一个最简单的UNET
- [ ] 到底什么是正确的提交格式啊啊啊啊啊啊啊啊啊！
- [ ] 加分支分类器预测时间（已做好dataloader返回）
- [ ] 搞清楚那个把序回归转为具体降水值的公式！
- validate.py里有几个TODO是我想不明白的问题，有空看看TAT

 
## 学长的建议
- 先跑出一个简单baseline
- OD加上降水概率
- 加一个分支直接同时预测降水和气温
- 加入时间监督信息，即加一个额外分类器层预测是几点的降水/气温！！及其重要！
- 一个人调loss一个人调模型
- 试试图网络(不用试了)
- （试过图网络后）有余力的话可以试试可形变卷积

## BEST记录
confidence lr=1e-4  100k降
encoder lr=1e-2 500k降
ODR 无预训练 lr=1e-3 50k降

## 验证记录
confidence2.pth(没用), encoderwithodr2.pth, decoder.pth, odr2.pth | ts:0.07359,0.01593,0.00697,0
confidence2.pth(没用), encoderwithodr3.pth, decoder.pth, odr3.pth | ts:0.16698,0.08479,0.00267,0
## The correlation matrices between the features are as follows:
![](corrMatrix.jpg)
