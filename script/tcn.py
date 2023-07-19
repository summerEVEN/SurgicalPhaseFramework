"""
            tcn_action.train(opt, model, train_dataset, test_dataset, device)
            # (opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/resnet_lstm")
"""

import os
import utils.labels
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from script.data_propre import dataset_propre
from utils.tensorboard_log import writer_log

import copy
import time
import pickle
from tqdm import tqdm

__all__ = ['train', 'test', "extract"]

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

def train(opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/tcn", debug = True):
    """
    数据处理的流程：
        1. 把整个视频输入到模型里面，前向传递，然后预测正确率
        （重点是得到一整个视频的特征，和一整个视频的label值）
    """
    model.to(device)

    criterion_phase = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)

    sequence_length = opt.sequence_length

    with open(opt.train_feature_path, 'rb') as f:
        g_LFB_train = pickle.load(f)

    with open(opt.test_feature_path, 'rb') as f:
        g_LFB_val = pickle.load(f)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_num_each_video = train_dataset.get_num_each_video()

    for epoch in range(opt.epoch):
        with tqdm(total=len(train_dataset), desc=f"Epoch {epoch+1}", unit="batch") as progress_bar:
            video_phase_count = 0
            for i, video_num in enumerate(train_num_each_video):
                
                labels_phase = []
                for j in range(video_phase_count, video_phase_count + video_num):
                    labels_phase.append(train_dataset[j][1])

                torch.cuda.empty_cache()

                # Sets the module in training mode.
                model.train()
                total = 0
                train_loss_phase = 0.0
                train_corrects_phase = 0
                batch_progress = 0.0
                running_loss_phase = 0.0
                minibatch_correct_phase = 0.0
                train_start_time = time.time()

                optimizer.zero_grad()
                print(labels_phase)
                labels_phase = labels_phase.to(device)

                # labels_phase = labels_phase[(sequence_length - 1)::sequence_length]


                long_feature = get_long_feature(start_index=video_phase_count,
                                        lfb=g_LFB_train, LFB_length=video_num)
                long_feature = (torch.Tensor(long_feature)).to(device)
                video_fe = long_feature.transpose(2, 1)

                outputs_phase = model.forward(video_fe)
                stages = outputs_phase.shape[0]

                _, preds_phase = torch.max(outputs_phase.data, 1)
                loss_phase = criterion_phase(outputs_phase, labels_phase)

                loss = loss_phase
                loss.backward()
                optimizer.step()

                running_loss_phase += loss_phase.data.item()
                train_loss_phase += loss_phase.data.item()

                batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
                train_corrects_phase += batch_corrects_phase
                total += len(labels_phase.data)

                progress_bar.update(video_num)
                # ！！！！
                video_phase_count = video_phase_count + video_num
        progress_bar.close()


        train_elapsed_time = time.time() - train_start_time

        epoch_acc = train_corrects_phase / total
        epoch_loss = train_loss_phase / total
        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, epoch_acc, epoch_loss))

        """
        保存当前最优秀的模型
        """
        debug = False
        if debug:
            acc = test(opt, model, test_dataset, device)
            if(acc > best_accuracy):
                best_epoch = epoch
                best_accuracy = acc
                torch.save(model.state_dict(), save_dir + '/{}-{}.model'.format(best_epoch, round(best_accuracy.item(), 4)))
        
    print("train success!")

def test(opt, model, test_dataset, device):
    model.to(device)
    model.eval()

    test_num_each_video = test_dataset.get_num_each_video()

    with open(opt.test_feature_path, 'rb') as f:
        g_LFB_val = pickle.load(f)

    with torch.no_grad():
        with tqdm(total=len(test_dataset), desc="test", unit="batch") as progress_bar:
            correct = 0
            total = 0
            video_phase_count = 0
            for i, video_num in enumerate(test_num_each_video):
                
                labels_phase = []
                for j in range(video_phase_count, video_phase_count + video_num):
                    labels_phase.append(test_dataset[j][1])
                
                long_feature = get_long_feature(start_index=video_phase_count,
                                        lfb=g_LFB_val, LFB_length=video_num)
                long_feature = (torch.Tensor(long_feature)).to(device)
                video_fe = long_feature.transpose(2, 1)

                outputs_phase = model.forward(video_fe)

                _, preds_phase = torch.max(outputs_phase.data, 1)

                correct += torch.sum(preds_phase == labels_phase.data)
                total += len(labels_phase.data)
                progress_bar.update(video_num)

                progress_bar.update(video_num)
                # ！！！！
                video_phase_count = video_phase_count + video_num
        progress_bar.close()
    print('Test: Acc {}'.format(correct / total))
    acc = correct / total
    return acc



def extract():

    print("Feature ectract success!")