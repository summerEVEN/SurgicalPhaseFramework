"""
这里是使用 resnet_lstm 网络进行训练，测试，验证，生成特征的几个函数的集合
使用这个网络的时候，数据需要进行特殊的处理，所以先尝试着写一下。

关于数据的处理:
这里牵扯到一个 batch_size 和 length 的关系（提前验证可以及时修改参数）

"""

"""
关于 tqdm 手动更新进度条存在一点点的小误差，由于视频的 sequence_length 引起的，暂时不去管吧hhh

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
    # sequence_length = opt.sequence_length
    # # 获取测试集每个视频包含的图片数的list
    # train_num_each = train_dataset.num_each_video()
    # train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    # # 训练集中有效的视频片段数量
    # train_slice_num = len(train_useful_start_idx)
    # np.random.shuffle(train_useful_start_idx)
    # train_idx_80 = []
    # for i in range(train_slice_num):
    #     for j in range(sequence_length):
    #         train_idx_80.append(train_useful_start_idx[i] + j)

    # train_loader_80 = DataLoader(
    #     train_dataset,
    #     batch_size=opt.batch_size,
    #     sampler=SeqSampler(train_dataset, train_idx_80),
    #     num_workers=opt.workers,
    #     pin_memory=False
    # )

    ######################## 训练集的数据id处理完成

    model.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion_phase = nn.CrossEntropyLoss(reduction='sum')

    learning_rate = opt.learning_rate
    sequence_length = opt.sequence_length

    optimizer = optim.Adam([
        {'params': model.share.parameters()},
        {'params': model.lstm.parameters(), 'lr': learning_rate},
        {'params': model.fc.parameters(), 'lr': learning_rate},
    ], lr=learning_rate / 10.0, weight_decay=1e-5)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0

    epochs = opt.epoch
    for epoch in range(epochs):
        with tqdm(total=len(train_dataset), desc=f"Epoch {epoch+1}", unit="batch") as progress_bar:
            train_loader = dataset_propre(opt, train_dataset, True)
            model.train()
            train_loss_phase = 0.0
            train_corrects_phase = 0
            total = 0
            batch_progress = 0.0
            running_loss_phase = 0.0
            train_start_time = time.time()

            for i, data in enumerate(train_loader):
                inputs, labels_phase = data[0].to(device), data[1].to(device)

                labels_phase = labels_phase[(sequence_length - 1)::sequence_length]

                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase = model.forward(inputs)
                # outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

                # print(outputs_phase.shape)
                # print(labels_phase.shape)

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

                progress_bar.update(len(labels_phase.data))
                # progress_bar.set_postfix({'Loss': loss_phase.data.item()})

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

def test(opt, model, test_dataset, device):
    """
    resnet_lstm
    """
    sequence_length = opt.sequence_length

    model.to(device)
    model.eval()
    test_loader = dataset_propre(opt, test_dataset)
    
    with torch.no_grad():

        correct = 0
        total = 0
        # process_bar = tqdm(total=len(test_dataset), desc=f"test_progress", unit="batch")
        with tqdm(total=len(test_dataset), desc="test", unit="batch") as progress_bar:
            for data in test_loader:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
                labels_phase = labels_phase[(sequence_length - 1)::sequence_length]
                inputs = inputs.view(-1, sequence_length, 3, 224, 224)
                outputs_phase = model.forward(inputs)
                # outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

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
    # import model.predictor.resnet_lstm as resnet_lstm
    # model = resnet_lstm.resnet_lstm_feature(opt)

    model.load_state_dict(torch.load(opt.model_path), strict=False)
    model.to(device)
    model.eval()

    print("-----------开始运行------------")

    train_loader = dataset_propre(opt, train_dataset)
    test_loader = dataset_propre(opt, test_dataset)

    # Long Term Feature bank
    g_LFB_train = np.zeros(shape=(0, 512))
    g_LFB_val = np.zeros(shape=(0, 512))

    with torch.no_grad():
        with tqdm(total=len(train_dataset), desc="train", unit="batch") as progress_bar:
            for data in train_loader:
                inputs, labels_phase = data[0].to(device), data[1].to(device)

                inputs = inputs.view(-1, opt.sequence_length, 3, 224, 224)
                outputs_feature = model.forward(inputs)

                for j in range(len(outputs_feature)):
                    save_feature = outputs_feature.data.cpu()[j].numpy()
                    # print(save_feature)
                    # 这里reshape的 512 和模型里面输出的大小是保持一致的
                    # 其他模型可能不一样，这里可以考虑之后，在config里面添加对应的参数，只写一个 extract函数hhh
                    save_feature = save_feature.reshape(1, 512)
                    g_LFB_train = np.concatenate((g_LFB_train, save_feature),axis=0)
                progress_bar.update(len(outputs_feature))
                # print("train feature length:",len(g_LFB_train))
            progress_bar.close()
            

        with tqdm(total=len(test_dataset), desc="test", unit="batch") as progress_bar:
            for data in test_loader:
                inputs, labels_phase = data[0].to(device), data[1].to(device)

                inputs = inputs.view(-1, opt.sequence_length, 3, 224, 224)
                outputs_feature = model.forward(inputs)

                for j in range(len(outputs_feature)):
                    save_feature = outputs_feature.data.cpu()[j].numpy()
                    save_feature = save_feature.reshape(1, 512)
                    g_LFB_val = np.concatenate((g_LFB_val, save_feature), axis=0)

                progress_bar.update(len(outputs_feature))
                # print("val feature length:",len(g_LFB_val))
            progress_bar.close()

    print("finish!")
    g_LFB_train = np.array(g_LFB_train)
    g_LFB_val = np.array(g_LFB_val)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, "g_LFB_train_st.pkl"), 'wb') as f:
        pickle.dump(g_LFB_train, f)

    with open(os.path.join(save_dir, "g_LFB_test_st.pkl"), 'wb') as f:
        pickle.dump(g_LFB_val, f)


if __name__ == "__main__":
    """
    UNIT TEST
    """

