# weather-rectify
Code for weather rectify competition


## TODO: 
- [ ] 训练三个网络
- [ ] ordinal网络trainer构建
- [ ] validate模块构建
- [ ] 跑测试集！

## 学长的建议
- 先跑出一个简单baseline
- OD加上降水概率
- 加一个分支直接同时预测降水和气温
- 加入时间监督信息，即加一个额外分类器层预测是几点的降水/气温！！及其重要！
- 一个人调loss一个人调模型
- 试试图网络
- （试过图网络后）有余力的话可以试试可形变卷积


## The correlation matrices between the features are as follows:
![](corrMatrix.jpg)
