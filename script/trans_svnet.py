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
大致过程：
1. 加载 resnet50 生成的空间特征
2. 加载训练好的 两层TCN 模型
3. (以整个视频为最小单元，需要整合每个视频的 label 信息，和 tcn feature 信息)
4. 整合单个视频的 label 信息，参考 tcn 模型即可
5. 整合 tcn feature 信息，先使用 get_long_feature 函数获得 tcn model 的输入，在使用 model 处理一下即可
6. …… 输入到 refinement model 咯
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

def train(opt, model, tcn_model, train_dataset, test_dataset, device, save_dir = "/result/model/trans_svnet", debug = True):
    model.train()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    # 获取TCN的模型的参数地址
    # 后面的许多参数,会逐个注释,然后放到 yml 配置文件里面
    import model.predictor.tcn as tcn
    tcn_model = tcn.MultiStageModel(opt)
    tcn_model = tcn_model.MultiStageModel(opt)
    tcn_model.load_state_dict(torch.load(opt.tcn_model_path), strict=False)
    tcn_model.to(device)
    tcn_model.eval()

    # 加载resnet50生成的空间特征
    # 训练集和测试集分别加载
    with open(opt.train_feature_path, 'rb') as f:
        g_LFB_train = pickle.load(f)

    with open(opt.test_feature_path, 'rb') as f:
        g_LFB_val = pickle.load(f)

    print("load completed")
    print("g_LFB_train shape:", g_LFB_train.shape)
    print("g_LFB_val shape:", g_LFB_val.shape)

    # 训练的时候，有把数据集（以视频为最小单元）乱序处理
    # 需要获得每个视频
    for epoch in range(opt.epoch):
        torch.cuda.empty_cache()
        random.shuffle(train_we_use_start_idx_80)

    # 记录训练最好的那个 epoch
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

        






