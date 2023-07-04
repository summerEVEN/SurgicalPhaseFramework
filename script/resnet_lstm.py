"""
这里是使用 resnet_lstm 网络进行训练，测试，验证，生成特征的几个函数的集合
使用这个网络的时候，数据需要进行特殊的处理，所以先尝试着写一下。

关于数据的处理:
这里牵扯到一个 batch_size 和 length 的关系（提前验证可以及时修改参数）

"""
import os
import utils.labels
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler

import copy
import time

__all__ = ['train', 'test', "extract"]

def get_useful_start_idx(sequence_length, list_each_length):
    """
    获取每个视频起始的图片id
    通过这个id, 可以快速在 dataset 里面，找到当前视频的每个视频片段的起始图片id
    """
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx

class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)
    
def dataset_propre(opt, dataset, shuffle = False):
    sequence_length = opt.sequence_length

    # 获取测试集每个视频包含的图片数的list
    dataset_num_each = []
    dataset_num_each = dataset.get_num_each_video()
    dataset_useful_start_idx = get_useful_start_idx(sequence_length, dataset_num_each)
    # 训练集中有效的视频片段数量
    dataset_slice_num = len(dataset_useful_start_idx)

    if shuffle:
        np.random.shuffle(dataset_useful_start_idx)

    dataset_idx = []
    for i in range(dataset_slice_num):
        for j in range(sequence_length):
            dataset_idx.append(dataset_useful_start_idx[i] + j)

    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=SeqSampler(dataset, dataset_idx),
        num_workers=opt.workers,
        pin_memory=False
    )

    return data_loader

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
    train_loader = dataset_propre(opt, train_dataset, True)
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
            outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

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
        for data in test_loader:
            inputs, labels_phase = data[0].to(device), data[1].to(device)
            labels_phase = labels_phase[(sequence_length - 1)::sequence_length]
            inputs = inputs.view(-1, sequence_length, 3, 224, 224)
            outputs_phase = model.forward(inputs)
            outputs_phase = outputs_phase[sequence_length - 1::sequence_length]

            _, preds_phase = torch.max(outputs_phase.data, 1)

            correct += torch.sum(preds_phase == labels_phase.data)
            total +=  len(labels_phase.data)
        print('Test: Acc {}'.format(correct / total))
    acc = correct / total
    return acc


def extract(opt, model, dataset, devices):
    """
    使用 resnet_lstm 网络提取视频特征
    """



if __name__ == "__main__":
    """
    UNIT TEST
    """

