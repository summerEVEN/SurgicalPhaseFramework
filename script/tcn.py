"""
目前对TCN的网络的结构,还是不太清楚,所以这部分代码修改得不太理想,糟糕

好家伙，得注意使用resnet50生成的特征的视频数，如果不存在那么多特征，后续的运行会出错
"""

import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from utils.ribbon import visualize_predictions_and_ground_truth

import copy
import time
import pickle
from tqdm import tqdm

__all__ = ['train']

def get_long_feature(start_index, lfb, LFB_length):
    # print("start_index: ", start_index, "length: ", LFB_length)
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        # print(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature

def train_frame_wise(opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/tcn", debug = True):
    """
    这个函数处理的特征是逐帧的特征，所有视频的特征都存在一个文件里
    """
    """
    数据处理的流程：
        1. 把整个视频输入到模型里面，前向传递，然后预测正确率
        （重点是得到一整个视频的特征，和一整个视频的label值）
        2. 
    """
    model.to(device)

    # criterion_phase = nn.CrossEntropyLoss(reduction='sum')
    weights_train = np.asarray([1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618,])
    criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device))
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)

    with open(opt.train_feature_path, 'rb') as f:
        g_LFB_train = pickle.load(f)
        print(g_LFB_train.shape)

    with open(opt.test_feature_path, 'rb') as f:
        g_LFB_val = pickle.load(f)
        print(g_LFB_val.shape)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获得每个视频的图片数(得到的是一个 list 数据)
    train_num_each_video = train_dataset.get_num_each_video()
    print("train_num_each_video: ", train_num_each_video)

    for epoch in range(opt.epoch):
        with tqdm(total=len(train_dataset), desc=f"Epoch {epoch+1}", unit="batch") as progress_bar:
            video_phase_count = 0

            model.train()
            total = 0
            train_loss_phase = 0.0
            train_corrects_phase = 0
            running_loss_phase = 0.0
            minibatch_correct_phase = 0.0
            train_start_time = time.time()
            for i, video_num in enumerate(train_num_each_video):
                optimizer.zero_grad()
                
                labels_phase = []
                for j in range(video_phase_count, video_phase_count + video_num - opt.sequence_length):
                    labels_phase.append(train_dataset[j][1])

                # Sets the module in training mode.

                # labels_phase = torch.Tensor(np.array(labels_phase))
                labels_phase = torch.LongTensor(np.array(labels_phase))
                labels_phase = labels_phase.to(device)

                long_feature = get_long_feature(start_index=video_phase_count,
                                        lfb=g_LFB_train, LFB_length=video_num - opt.sequence_length)
                long_feature = (torch.Tensor(long_feature)).to(device)
                video_fe = long_feature.transpose(2, 1)

                outputs_phase = model.forward(video_fe)
                # print("outputs_phase.shape: ", outputs_phase.shape)
                stages = outputs_phase.shape[0]


                clc_loss = 0
                for j in range(stages):  ### make the interuption free stronge the more layers.
                    p_classes = []
                    p_classes = outputs_phase[j].squeeze().transpose(1, 0)
                    ce_loss = criterion_phase(p_classes, labels_phase)
                    clc_loss += ce_loss
                clc_loss = clc_loss / (stages * 1.0)

                _, preds_phase = torch.max(outputs_phase[stages-1].squeeze().transpose(1, 0).data, 1)

                loss = clc_loss
                # print(loss)
                loss.backward()
                optimizer.step()

                running_loss_phase += clc_loss.data.item()
                train_loss_phase += clc_loss.data.item()

                batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
                train_corrects_phase += batch_corrects_phase
                total += len(labels_phase.data)

                progress_bar.update(video_num)
                video_phase_count = video_phase_count + video_num
        progress_bar.close()

        train_elapsed_time = time.time() - train_start_time

        epoch_acc = train_corrects_phase / total
        epoch_loss = train_loss_phase / total
        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, epoch_acc, epoch_loss))

        """
        保存当前最优秀的模型
        """
        if debug:
            acc = test_frame_wise(opt, model, test_dataset, device)
            if(acc > best_accuracy):
                best_epoch = epoch
                best_accuracy = acc
                torch.save(model.state_dict(), save_dir + '/{}-{}.model'.format(best_epoch, round(best_accuracy.item(), 4)))
        
    print("train success!")

def test_frame_wise(opt, model, test_dataset, device):
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
                labels_phase = torch.LongTensor(np.array(labels_phase))
                labels_phase = labels_phase.to(device)
                
                long_feature = get_long_feature(start_index=video_phase_count,
                                        lfb=g_LFB_val, LFB_length=video_num)
                long_feature = (torch.Tensor(long_feature)).to(device)
                video_fe = long_feature.transpose(2, 1)

                outputs_phase = model.forward(video_fe)
                stages = outputs_phase.shape[0]

                # _, preds_phase = torch.max(outputs_phase.data, 1)
                _, preds_phase = torch.max(outputs_phase[stages-1].squeeze().transpose(1, 0).data, 1)

                # print(preds_phase.is_cuda)
                # print(labels_phase.is_cuda)

                correct += torch.sum(preds_phase == labels_phase.data)
                total += len(labels_phase.data)
                progress_bar.update(video_num)
                # ！！！！
                video_phase_count = video_phase_count + video_num
        progress_bar.close()
    print('Test: Acc {}'.format(correct / total))
    acc = correct / total
    return acc

def train_video(opt, model, train_loader, test_loader, device, save_dir = "./result/model/tcn_video", debug = True):
    model.to(device)
    weights_train = np.asarray([1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618,])
    criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device))
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)

    best_accuracy = 0.0
    best_epoch = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(opt.epoch):
        model.train()
        total = 0
        train_loss_phase = 0.0
        train_corrects_phase = 0
        running_loss_phase = 0.0

        for (video, labels, video_name) in tqdm(train_loader):
            labels = torch.Tensor(labels).long()        
            video, labels = video.float().to(device), labels.to(device) 

            # print("video.shape: ", video[0][0].shape, video[0][0].data)
            video_fe = video.transpose(2, 1)
            outputs_phase = model.forward(video_fe)
            # print("outputs_phase.shape: ", outputs_phase.shape)
            stages = outputs_phase.shape[0]

            clc_loss = 0
            for j in range(stages):  ### make the interuption free stronge the more layers.
                p_classes = []
                p_classes = outputs_phase[j].squeeze().transpose(1, 0)
                ce_loss = criterion_phase(p_classes, labels)
                clc_loss += ce_loss
            clc_loss = clc_loss / (stages * 1.0)

            _, preds_phase = torch.max(outputs_phase[stages-1].squeeze().transpose(1, 0).data, 1)

            loss = clc_loss
            loss.backward()
            optimizer.step()

            running_loss_phase += clc_loss.data.item()
            train_loss_phase += clc_loss.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels.data)
            train_corrects_phase += batch_corrects_phase
            total += len(labels.data)

        epoch_acc = train_corrects_phase / total
        epoch_loss = train_loss_phase / total
        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, epoch_acc, epoch_loss))

        """
        保存当前最优秀的模型
        """
        if debug:
            acc = test_video(opt, model, test_loader, device)
            if(acc > best_accuracy):
                best_epoch = epoch
                best_accuracy = acc
                torch.save(model.state_dict(), save_dir + '/{}-{}.model'.format(best_epoch, round(best_accuracy.item(), 4)))
        
    print("train success!")

def test_video(opt, model, test_loader, device):
    model.to(device)
    model.eval()

    total = 0
    test_corrects_phase = 0

    for (video, labels, video_name) in tqdm(test_loader):
        labels = torch.Tensor(labels).long()        
        video, labels = video.float().to(device), labels.to(device) 

        video_fe = video.transpose(2, 1)
        outputs_phase = model.forward(video_fe)
        stages = outputs_phase.shape[0]

        _, preds_phase = torch.max(outputs_phase[stages-1].squeeze().transpose(1, 0).data, 1)

        batch_corrects_phase = torch.sum(preds_phase == labels.data)
        test_corrects_phase += batch_corrects_phase
        total += len(labels.data)

    epoch_acc = test_corrects_phase / total
    print('test Acc {}'.format(epoch_acc))
    return epoch_acc

def frame_wise_visualization(opt, model, test_dataset, device):
    model.load_state_dict(torch.load(opt.eval_model_path), strict=False)
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
                labels_phase = torch.LongTensor(np.array(labels_phase))
                labels_phase = labels_phase.to(device)
                
                long_feature = get_long_feature(start_index=video_phase_count,
                                        lfb=g_LFB_val, LFB_length=video_num)
                long_feature = (torch.Tensor(long_feature)).to(device)
                video_fe = long_feature.transpose(2, 1)

                outputs_phase = model.forward(video_fe)
                stages = outputs_phase.shape[0]

                # _, preds_phase = torch.max(outputs_phase.data, 1)
                _, preds_phase = torch.max(outputs_phase[stages-1].squeeze().transpose(1, 0).data, 1)

                # print(preds_phase.is_cuda)
                # print(labels_phase.is_cuda)

                batch_correct = torch.sum(preds_phase == labels_phase.data)
                correct += batch_correct
                total += len(labels_phase.data)

                img_path = test_dataset[video_phase_count][3]
                video_name = os.path.split(os.path.split(img_path)[0])[1]

                visualize_predictions_and_ground_truth(preds_phase, labels_phase, int(batch_correct.data)/len(labels_phase.data), 
                                                       video_name, opt.model_name, save_dir='./result/visualization/')

                progress_bar.update(video_num)
                video_phase_count = video_phase_count + video_num
        progress_bar.close()
    print('Test: Acc {}'.format(correct / total))
    acc = correct / total
    return acc

def video_visualization(opt, model, test_loader, device):
    model.load_state_dict(torch.load(opt.eval_model_path), strict=False)
    model.to(device)
    model.eval()
    total = 0
    test_corrects_phase = 0

    for (video, labels, video_name) in tqdm(test_loader):
        labels = torch.Tensor(labels).long()        
        video, labels = video.float().to(device), labels.to(device) 

        video_fe = video.transpose(2, 1)
        outputs_phase = model.forward(video_fe)
        stages = outputs_phase.shape[0]

        _, preds_phase = torch.max(outputs_phase[stages-1].squeeze().transpose(1, 0).data, 1)

        batch_corrects_phase = torch.sum(preds_phase == labels.data)
        test_corrects_phase += batch_corrects_phase
        total += len(labels.data)

        visualize_predictions_and_ground_truth(preds_phase, labels, int(batch_corrects_phase.data)/len(labels.data), 
                                               video_name, opt.model_name, save_dir='./result/visualization/')

    epoch_acc = test_corrects_phase / total
    print('test Acc {}'.format(epoch_acc))
    return epoch_acc


def extract_video(opt, model, train_loader, test_loader, device):
    """
    提取特征
    """
    print("TODO")
    return 0
    model.load_state_dict(torch.load(opt.eval_model_path), strict=False)
    model.to(device)
    model.eval()
    total = 0
    test_corrects_phase = 0

    for (video, labels, video_name) in tqdm(test_loader):
        labels = torch.Tensor(labels).long()        
        video, labels = video.float().to(device), labels.to(device) 

        video_fe = video.transpose(2, 1)
        outputs_phase = model.forward(video_fe)
        stages = outputs_phase.shape[0]

        video_feature, preds_phase = torch.max(outputs_phase[stages-1].squeeze().transpose(1, 0).data, 1)

        # np.save(os.path.join(test_save_dir, video_name + ".npy"), video_feature_x)

        batch_corrects_phase = torch.sum(preds_phase == labels.data)
        test_corrects_phase += batch_corrects_phase
        total += len(labels.data)

    epoch_acc = test_corrects_phase / total
    print('test Acc {}'.format(epoch_acc))
    return epoch_acc