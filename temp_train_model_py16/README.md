- 仅需运行main.py即可训练（需要tqdm）
> 在mymodel第19行，设置了isFirstTime=True,则将会花一些时间对训练输入数据扫描计算mean和std，进行归一化。
> 在processed_data/ 中有扫描初赛1962个数据生成的mean和std，可以通过设置isFirstTime=False来快速开始训练