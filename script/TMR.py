import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler

from script.data_propre import dataset_propre, get_dict_start_idx_LFB

import copy
import time
import pickle
import os

__all__ = ['train', 'test', "extract"]

"""
            print(" TMR 开始加载 ")
            import model.refinement.TMR as TMR_model
            model = TMR_model.resnet_lstm(opt)
            print(" TMR 加载成功 ！！！！")
"""
def get_long_feature(opt, start_index_list, dict_start_idx_LFB, lfb):
    long_feature = []
    for j in range(len(start_index_list)):
        long_feature_each = []
        
        # 上一个存在feature的index
        last_LFB_index_no_empty = dict_start_idx_LFB[int(start_index_list[j])]
        
        for k in range(opt.LFB_length):
            LFB_index = (start_index_list[j] - k - 1)
            if int(LFB_index) in dict_start_idx_LFB:                
                LFB_index = dict_start_idx_LFB[int(LFB_index)]
                long_feature_each.append(lfb[LFB_index])
                last_LFB_index_no_empty = LFB_index
            else:
                long_feature_each.append(lfb[last_LFB_index_no_empty])
            
        long_feature.append(long_feature_each)
    return long_feature

def train(opt, train_dataset, test_dataset, device, save_dir = "./result/model/TMR", debug = True):
    """
    TMR 模型的训练函数
    1. 实例化模型，加载前面训练好的 resnet_lstm 模型的参数
    """
    import model.refinement.TMR as TMR_model
    model = TMR_model.resnet_lstm(opt)
    model.load_state_dict(torch.load(opt.model_path), strict=False)
    model.to(device)

    learning_rate = opt.learning_rate
    sequence_length = opt.sequence_length
    train_feature_path = opt.train_feature_path
    test_feature_path = opt.test_feature_path
    model_path = opt.model_path
    epochs = opt.epoch

    dict_train_start_idx_LFB = get_dict_start_idx_LFB(sequence_length, train_dataset)

    train_loader = dataset_propre(opt, train_dataset, True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion_phase = nn.CrossEntropyLoss(reduction='sum')

    print("loading features!>.........")

    with open(train_feature_path, 'rb') as f:
        g_LFB_train = pickle.load(f)

    with open(test_feature_path, 'rb') as f:
        g_LFB_val = pickle.load(f)

    print("g_LFB_train shape:",g_LFB_train.shape)
    print("g_LFB_val shape:",g_LFB_val.shape)

    torch.cuda.empty_cache()

    optimizer = optim.Adam([
                {'params': model.share.parameters()},
                {'params': model.lstm.parameters()},
                {'params': model.time_conv.parameters(), 'lr': learning_rate},
                {'params': model.nl_block.parameters(), 'lr': learning_rate},
                {'params': model.fc_h_c.parameters(), 'lr': learning_rate},
                {'params': model.fc_c.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10, weight_decay=1e-5)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        total = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        train_start_time = time.time()

        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels_phase = data[0].to(device), data[1].to(device)
            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

            start_index_list = data[2]
            start_index_list = start_index_list[0::sequence_length]
            long_feature = get_long_feature(opt,
                                            start_index_list=start_index_list,
                                            dict_start_idx_LFB=dict_train_start_idx_LFB,
                                            lfb=g_LFB_train)

            long_feature = (torch.Tensor(np.array(long_feature))).to(device)

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            # TODO
            outputs_phase = model.forward(inputs, long_feature=long_feature)

            _, preds_phase = torch.max(outputs_phase.data, 1)
            loss_phase = criterion_phase(outputs_phase, labels_phase)

            loss = loss_phase
            loss.backward()
            optimizer.step()

            running_loss_phase += loss_phase.data.item()
            train_loss_phase += loss_phase.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase

            # print(type(labels_phase.data))
            # print(len(labels_phase.data))
            total += len(labels_phase.data)
            # ------------- len(这里可能有问题)

        epoch_acc = train_corrects_phase / total
        epoch_loss = train_loss_phase / total
        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, epoch_acc, epoch_loss))
        if debug:
            acc = test(opt, model, test_dataset, device)
            if(acc > best_accuracy):
                best_epoch = epoch
                best_accuracy = acc
                torch.save(model.state_dict(), save_dir + '/{}-{}.model'.format(best_epoch, round(best_accuracy.item(), 4)))
        
    print("train success!")

def test(opt, model, test_dataset, device):
    sequence_length = opt.sequence_length

    model.to(device)
    model.eval()
    test_loader = dataset_propre(opt, test_dataset)
    dict_val_start_idx_LFB = get_dict_start_idx_LFB(sequence_length, test_dataset)

    with open(opt.test_feature_path, 'rb') as f:
        g_LFB_val = pickle.load(f)
    
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels_phase = data[0].to(device), data[1].to(device)
            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

            start_index_list = data[2]
            start_index_list = start_index_list[0::sequence_length]
            long_feature = get_long_feature(opt,
                                            start_index_list=start_index_list,
                                            dict_start_idx_LFB=dict_val_start_idx_LFB,
                                            lfb=g_LFB_val)

            long_feature = (torch.Tensor(np.array(long_feature))).to(device)

            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            # TODO
            outputs_phase = model.forward(inputs, long_feature=long_feature)
            inputs = inputs.view(-1, sequence_length, 3, 224, 224)

            _, preds_phase = torch.max(outputs_phase.data, 1)

            correct += torch.sum(preds_phase == labels_phase.data)
            total +=  len(labels_phase.data)
        print('Test: Acc {}'.format(correct / total))
    acc = correct / total
    return acc

