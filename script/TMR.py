import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler

from script.data_propre import dataset_propre, get_dict_start_idx_LFB
from utils.ribbon import visualize_predictions_and_ground_truth
import copy
import time
import pickle
import os
from tqdm import tqdm

__all__ = ['train', 'test', 'evaluate_and_visualize']

"""
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

def train(opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/TMR", debug = True):
    """
    TMR 模型的训练函数
    1. 实例化模型，加载前面训练好的 resnet_lstm 模型的参数
    2. 正常训练
    """
    model.load_state_dict(torch.load(opt.model_path), strict=False)
    model.to(device)

    learning_rate = opt.learning_rate
    sequence_length = opt.sequence_length
    train_feature_path = opt.train_feature_path
    val_feature_path = opt.val_feature_path
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

    with open(val_feature_path, 'rb') as f:
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
        with tqdm(total=len(train_dataset), desc=f"Epoch {epoch+1}", unit="batch") as progress_bar:
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

                total += len(labels_phase.data)
                progress_bar.update(len(labels_phase.data))

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

    with open(opt.val_feature_path, 'rb') as f:
        g_LFB_val = pickle.load(f)
    
    with torch.no_grad():
        with tqdm(total=len(test_dataset), desc="Test", unit="batch") as progress_bar:
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
                outputs_phase = model.forward(inputs, long_feature=long_feature)


                _, preds_phase = torch.max(outputs_phase.data, 1)

                correct += torch.sum(preds_phase == labels_phase.data)
                total +=  len(labels_phase.data)
                progress_bar.update(len(labels_phase.data))
        print('Test: Acc {}'.format(correct / total))
    acc = correct / total
    return acc

def evaluate_and_visualize(opt, model, test_dataset, device):
    """
    生成每个视频的可视化结果，以及每个视频预测的结果的txt文档
    1. 根据数据集里面的 get_num_each_video , 把每个视频的图片区分开（使用for循环处理）
    2. 在for循环里，拼接 pred_phase_v 和 label_phase_v,记录当前视频预测正确的图片数
    3. 
    """
    sequence_length = opt.sequence_length
    model.load_state_dict(torch.load(opt.eval_model_path), strict=False)
    model.to(device)
    model.eval()
    
    test_loader = dataset_propre(opt, test_dataset)
    dict_val_start_idx_LFB = get_dict_start_idx_LFB(sequence_length, test_dataset)

    with open(opt.val_feature_path, 'rb') as f:
        g_LFB_val = pickle.load(f)
    
    with torch.no_grad():
        with tqdm(total=len(test_dataset), desc="Test", unit="batch") as progress_bar:
            correct = 0
            total = 0

            # 用来记录所有视频的预测结果和groun_turth的值
            all_pred_phase = []
            all_label_phase = []
            
            # 获取每个视频的图片数，然后减去序列长度，得到每个视频的视频片段数
            test_num_each_video = test_dataset.get_num_each_video()
            test_clip_each_video = [x - sequence_length for x in test_num_each_video]

            # 记录当前处理的视频数
            video_processed_num = 0

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
                outputs_phase = model.forward(inputs, long_feature=long_feature)


                _, preds_phase = torch.max(outputs_phase.data, 1)
                """
                把 preds_phase 按照视频的帧数拼接，然后保存到一个txt文件里面
                同时生成和 ground_turth 的对比图片，下面带上这个图片的正确率
                """
                # visualize_predictions_and_ground_truth(preds_phase_v, labels_phase_v, acc, video_num, opt.model_name, save_dir='./result/visualization/')

                all_pred_phase.extend(preds_phase.tolist())
                all_label_phase.extend(labels_phase.tolist())

                if(len(all_label_phase) >= test_clip_each_video[video_processed_num]):
                    # 如果符合判断条件，说明当前视频的所有clip处理完成，可以生成当前视频的预测结果

                    count_same_elements = sum(i == j for i, j in zip(all_pred_phase, all_label_phase))
                    img_path = data[3][0]
                    video_name = os.path.split(os.path.split(img_path)[0])[1]
                    visualize_predictions_and_ground_truth(all_pred_phase[: test_clip_each_video[video_processed_num]], 
                                                           all_label_phase[: test_clip_each_video[video_processed_num]], 
                                                           count_same_elements/test_clip_each_video[video_processed_num], 
                                                           video_name, opt.model_name, save_dir='./result/visualization/')
                    
                    all_pred_phase = all_pred_phase[test_clip_each_video[video_processed_num] :]
                    all_label_phase = all_label_phase[test_clip_each_video[video_processed_num] :]
                    video_processed_num = video_processed_num + 1

                correct += torch.sum(preds_phase == labels_phase.data)
                total +=  len(labels_phase.data)
                progress_bar.update(len(labels_phase.data))
        print('Test: Acc {}'.format(correct / total))
    acc = correct / total
    return acc

