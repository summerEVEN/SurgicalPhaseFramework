"""
这里是使用 resnet_lstm 网络进行训练，测试，验证，生成特征的几个函数的集合

关于数据的处理:
这里牵扯到一个 batch_size 和 length 的关系：
batch_size 必须是 length 的倍数
（如果添加提前验证的代码，可以在训练前就结束运行及时修改参数）

"""

"""
关于 tqdm 手动更新进度条存在一点点的小误差，
由于视频的 sequence_length 引起的，暂时不管
后面找找更加方便的实现方法
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

import copy
import time
import pickle
from tqdm import tqdm

__all__ = ['train', 'test', "extract"]


def train(opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/resnet_lstm", debug=True):
    """
    resnet_lstm 的训练步骤：
    1. 生成长度为 sequence length 的视频片段 （像原来那样，考虑每个视频的
    2. 模型训练 （保存训练的数值）
    3. 模型验证 （保存训练的数值）
    4. 保存最优模型的参数
    """
    model.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion_phase = nn.CrossEntropyLoss(reduction='sum')

    learning_rate = opt.learning_rate
    sequence_length = opt.sequence_length
    epochs = opt.epoch

    optimizer = optim.Adam([
        {'params': model.share.parameters()},
        {'params': model.lstm.parameters(), 'lr': learning_rate},
        {'params': model.fc.parameters(), 'lr': learning_rate},
    ], lr=learning_rate / 10.0, weight_decay=1e-5)


    best_accuracy = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        with tqdm(total=len(train_dataset), desc=f"train epoch {epoch+1}", unit="batch") as progress_bar:
            train_loader = dataset_propre(opt, train_dataset, True)
            model.train()

            train_loss_phase = 0.0
            train_corrects_phase = 0
            total = 0
            running_loss_phase = 0.0

            for i, data in enumerate(train_loader):
                inputs, labels_phase = data[0].to(device), data[1].to(device)

                labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase = model.forward(inputs)

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

        """
        保存当前最优秀的模型
        """
        if debug:
            acc = test(opt, model, test_dataset, device)
            if(acc > best_accuracy):
                best_epoch = epoch
                best_accuracy = acc
                torch.save(model.state_dict(), save_dir + '/{}-{}.model'.format(best_epoch, round(best_accuracy.item(), 4)))
        
    print("train success!")

def test(opt, model, test_dataset, device, dir_res="./result/label/resnet_lstm"):
    sequence_length = opt.sequence_length

    model.to(device)
    model.eval()
    test_loader = dataset_propre(opt, test_dataset)
    
    with torch.no_grad():
        correct = 0
        total = 0

        with tqdm(total=len(test_dataset), desc="test", unit="batch") as progress_bar:
            for data in test_loader:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
                labels_phase = labels_phase[(sequence_length - 1)::sequence_length]
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase = model.forward(inputs)

                _, preds_phase = torch.max(outputs_phase.data, 1)

                correct += torch.sum(preds_phase == labels_phase.data)
                total +=  len(labels_phase.data)

                progress_bar.update(len(labels_phase.data))
    print('Test: Acc {}'.format(correct / total))
    acc = correct / total
    return acc


def extract(opt, model, train_dataset, test_dataset, device, save_dir = "./result/feature/resnet_lstm"):
    """
    使用 resnet_lstm 网络提取视频特征
    """

    model.load_state_dict(torch.load(opt.model_path), strict=False)
    model.to(device)
    model.eval()

    print("-----------extract------------")

    train_loader = dataset_propre(opt, train_dataset)
    test_loader = dataset_propre(opt, test_dataset)

    # Long Term Feature bank
    # 这里的 512 和模型的输出特征大小保持一致
    g_LFB_train = np.zeros(shape=(0, 512))
    g_LFB_val = np.zeros(shape=(0, 512))

    with torch.no_grad():
        with tqdm(total=len(train_dataset), desc="train", unit="batch") as progress_bar:
            for data in train_loader:
                inputs = data[0].to(device)

                inputs = inputs.view(-1, opt.sequence_length, 3, 224, 224)
                outputs_feature = model.forward(inputs)
                g_LFB_train = np.concatenate((g_LFB_train, outputs_feature),axis=0)

                progress_bar.update(len(outputs_feature))
            progress_bar.close()       

        with tqdm(total=len(test_dataset), desc="test", unit="batch") as progress_bar:
            for data in test_loader:
                inputs = data[0].to(device)

                inputs = inputs.view(-1, opt.sequence_length, 3, 224, 224)
                outputs_feature = model.forward(inputs)

                g_LFB_val = np.concatenate((g_LFB_val, outputs_feature), axis=0)

                progress_bar.update(len(outputs_feature))
                # print("val feature length:",len(g_LFB_val))
            progress_bar.close()

    print("finish!")
    g_LFB_train = np.array(g_LFB_train)
    g_LFB_val = np.array(g_LFB_val)

    print("len(g_LFB_train): ", len(g_LFB_train))
    print("len(g_LFB_val): ", len(g_LFB_val))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "g_LFB_train_st.pkl"), 'wb') as f:
        pickle.dump(g_LFB_train, f)

    with open(os.path.join(save_dir, "g_LFB_test_st.pkl"), 'wb') as f:
        pickle.dump(g_LFB_val, f)

def extract_video(opt, model, train_dataset, test_dataset, device, save_dir = "./result/feature_video/resnet_lstm"):

    sequence_length = opt.sequence_length
    model.load_state_dict(torch.load(opt.eval_model_path), strict=False)
    model.to(device)
    model.eval()

    if  not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    train_loader = dataset_propre(opt, train_dataset)
    test_loader = dataset_propre(opt, test_dataset)
    
    with torch.no_grad():
        with tqdm(total=len(train_dataset), desc="extract train feature", unit="batch") as progress_bar:
            video_feature_train = np.zeros(shape=(0, 512))
            
            # 获取每个视频的图片数，然后减去序列长度 + 1，得到每个视频的视频片段数
            train_num_each_video = train_dataset.get_num_each_video()
            train_clip_each_video = [x - sequence_length + 1 for x in train_num_each_video]

            # print("train_clip_each_video : ", train_clip_each_video )

            # 记录当前已经处理到第几个视频了
            video_processed_num = 0

            for data in train_loader:
                inputs = data[0].to(device)

                start_index_list = data[2]
                start_index_list = start_index_list[0::sequence_length]

                # 244 * 244 这里是固定的
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase = model.forward(inputs)
                video_feature_train = np.concatenate((video_feature_train, outputs_phase.cpu()), axis=0)

                if(video_feature_train.shape[0] >= train_clip_each_video[video_processed_num]):
                    # 如果符合判断条件，说明当前视频的所有clip处理完成，可以生成当前视频的预测结果

                    img_path = data[3][0]
                    # print("img_path: ", img_path)
                    # print("start_index_list: ", start_index_list)
                    video_name = os.path.split(os.path.split(img_path)[0])[1]

                    
                    video_feature_x = video_feature_train[: train_clip_each_video[video_processed_num]]
                    np.save(os.path.join(save_dir, video_name + ".npy"), video_feature_x)

                    # print("video_feature_x: ", video_feature_x.shape)

                    video_feature_train = video_feature_train[train_clip_each_video[video_processed_num] :]
                    video_processed_num = video_processed_num + 1

                    # print("第{}个视频特征处理完成，采样率为：{}".format(video_processed_num, opt.sample_rate))


                progress_bar.update(len(outputs_phase.data))
            print("train part success!!")

    with torch.no_grad():
        with tqdm(total=len(test_dataset), desc="extract test feature", unit="batch") as progress_bar:
            video_feature_test = np.zeros(shape=(0, 512))
            test_num_each_video = test_dataset.get_num_each_video()
            test_clip_each_video = [x - sequence_length + 1 for x in test_num_each_video]

            video_processed_num = 0

            for data in test_loader:
                inputs = data[0].to(device)

                start_index_list = data[2]
                start_index_list = start_index_list[0::sequence_length]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase = model.forward(inputs)
                video_feature_test = np.concatenate((video_feature_test, outputs_phase.cpu()), axis=0)

                if(video_feature_test.shape[0] >= test_clip_each_video[video_processed_num]):
                    # 如果符合判断条件，说明当前视频的所有clip处理完成，可以生成当前视频的预测结果
                    img_path = data[3][0]
                    video_name = os.path.split(os.path.split(img_path)[0])[1]

                    video_feature_x = video_feature_test[: test_clip_each_video[video_processed_num]]
                    np.save(os.path.join(save_dir, video_name + ".npy"), video_feature_x)

                    video_feature_test = video_feature_test[test_clip_each_video[video_processed_num] :]
                    video_processed_num = video_processed_num + 1
                progress_bar.update(len(outputs_phase.data))
            print("test part success!!")

if __name__ == "__main__":
    """
    UNIT TEST
    """

