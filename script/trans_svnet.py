import torch
from torch import optim
from torch import nn
import numpy as np
import pickle, time
import random
from sklearn import metrics
import copy
import os, subprocess

from script.data_propre import dataset_propre
import model.predictor.tcn as tcn_model

"""

"""




def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature

def train(opt, model, train_dataset, test_dataset, device, save_dir = "/result/model/trans_svnet", debug = True):
    model.train()
    model.to(device)

    # 获取TCN的模型的参数地址
    # model = mstcn.MultiStageModel(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)
    # 后面的许多参数,会逐个注释,然后放到 yml 配置文件里面
    tcn_model = tcn_model.MultiStageModel(opt)
    tcn_model.load_state_dict(torch.load(opt.tcn_model_path), strict=False)

    # 加载resnet50生成的空间特征
    # 训练集和测试集分别加载
    with open(opt.train_, 'rb') as f:
        g_LFB_train = pickle.load(f)

    with open("./LFB/g_LFB50_val.pkl", 'rb') as f:
        g_LFB_val = pickle.load(f)

    with open("./LFB/g_LFB50_test.pkl", 'rb') as f:
        g_LFB_test = pickle.load(f)

    print("load completed")

    print("g_LFB_train shape:", g_LFB_train.shape)
    print("g_LFB_val shape:", g_LFB_val.shape)

    # 输入的数据预处理
    # 


    # 记录训练最好的那个 epoch
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    for epoch in range(opt.epochs):
        torch.cuda.empty_cache()
        train_loader = dataset_propre(opt, train_dataset, True)






