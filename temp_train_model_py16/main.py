import yaml

if __name__ == "__main__":
    from mymodel import model
    max_epochs = 10000  #最大迭代周期数
    num_gpus = 1  #使用的GPU数目
    num_1 = 111  #每个周期内输出训练损失的迭代数间隔
    num_2 = 50  #输出验证集表现的周期间隔
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    device = 'cuda'
    args = dict()
    args['max_epochs'] = max_epochs
    args['num_gpus'] = num_gpus
    args['show_trainloss_every_num_iterations_per_epoch'] = num_1
    args['show_validperformance_every_num_epochs'] = num_2
    args['device'] = device
    args['config'] = config

    #model_inst = model('/mnt/pami23/stma/weather/train/', **args).to(device)
    model_inst = model(config['train_dir'], **args).to(device)
    #输 出train_data中 作 为 验 证 样 本 的 序 号
    #print(model_inst.valid_example)
    #输 出 训 练 损 失 函 数 和 验 证 评 价 函 数 名 称 ，
    #验 证 评 价 函 数 不 一 定 需 要 和 比 赛 评 价 指 标 一 致 。
    #print(model_inst.train_loss, model_inst.valid_perfomace)
    #开 始 训 练
    #model_inst.initialize('/mnt/pami23/zhengxin/projects/temp/checkpoint_new_timeloss/unet_lr0405_05816.pth')
    #model_inst.initialize('checkpoint/unet_out_59_best.pth')
    model_inst.train(lr=1e-4, save_path='checkpoint/unet_0222_58_spatial.pth')
