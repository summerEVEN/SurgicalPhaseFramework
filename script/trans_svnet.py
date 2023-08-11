import torch
from torch import optim
from torch import nn
import numpy as np
import pickle, time
import random
from sklearn import metrics
import copy
import os, subprocess
from tqdm import tqdm

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

def train(opt, model, train_dataset, test_dataset, device, save_dir = "/result/model/trans_svnet", debug = True):
    model.train()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion_phase = nn.CrossEntropyLoss()

    # 获取TCN的模型的参数地址
    # 后面的许多参数,会逐个注释,然后放到 yml 配置文件里面
    import model.predictor.tcn as tcn
    tcn_model = tcn.MultiStageModel(opt)
    # tcn_model = tcn_model.MultiStageModel(opt)
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

    # 获得每个视频的图片数(得到的是一个 list 数据)
    train_num_each_video = train_dataset.get_num_each_video()
    print("train_num_each_video: ", train_num_each_video)

    for epoch in range(opt.epoch):
        torch.cuda.empty_cache()
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
                # 
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

                out_features = tcn_model.forward(video_fe)[-1]
                out_features = out_features.squeeze(1)

                p_classes1 = model(out_features.detach(), long_feature)

                clc_loss = criterion_phase(p_classes1, labels_phase)

                _, preds_phase = torch.max(p_classes1.data, 1)

                loss = clc_loss
                #print(loss.data.cpu().numpy())
                loss.backward()
                optimizer.step()

                running_loss_phase += clc_loss.data.item()
                train_loss_phase += clc_loss.data.item()

                batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
                train_corrects_phase += batch_corrects_phase
                total += len(labels_phase.data)

                # # print("outputs_phase.shape: ", outputs_phase.shape)
                # stages = outputs_phase.shape[0]


                # clc_loss = 0
                # for j in range(stages):  ### make the interuption free stronge the more layers.
                #     p_classes = []
                #     p_classes = outputs_phase[j].squeeze().transpose(1, 0)
                #     ce_loss = criterion_phase(p_classes, labels_phase)
                #     clc_loss += ce_loss
                # clc_loss = clc_loss / (stages * 1.0)

                # _, preds_phase = torch.max(outputs_phase[stages-1].squeeze().transpose(1, 0).data, 1)

                # loss = clc_loss
                # # print(loss)
                # loss.backward()
                # optimizer.step()

                # running_loss_phase += clc_loss.data.item()
                # train_loss_phase += clc_loss.data.item()

                # batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
                # train_corrects_phase += batch_corrects_phase
                # total += len(labels_phase.data)

                progress_bar.update(video_num)
                video_phase_count = video_phase_count + video_num
        progress_bar.close()

        epoch_acc = train_corrects_phase / total
        epoch_loss = train_loss_phase / total
        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, epoch_acc, epoch_loss))

        debug = False
        if debug:
            acc = test(opt, model, test_dataset, device)
            if(acc > best_accuracy):
                best_epoch = epoch
                best_accuracy = acc
                torch.save(model.state_dict(), save_dir + '/{}-{}.model'.format(best_epoch, round(best_accuracy.item(), 4)))
        
    print("train success!")
    # 记录训练最好的那个 epoch
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

def test(opt, model, test_dataset, device):
    print("success!!")






