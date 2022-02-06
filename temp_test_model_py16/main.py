from mymodel import model

if __name__ == "__main__":
    device = 'cuda'
    example = 'data_test/example00001'
    model_inst = model().to(device)
    #preds_list是 一 个 长 度 为10的list
    #其 中 每 个 元 素 是 对 应 长 度 的 预 报 数 据(一 维numpy数 据)
    #如 ：preds_list[0].shape=(281,)
    pred_list = model_inst.forward(example)
    print(pred_list[0].shape)