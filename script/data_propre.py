
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
import pickle

__all__ = ["dataset_propre", "get_useful_start_idx", "get_dict_start_idx_LFB"]

def get_useful_start_idx(sequence_length, list_each_length):
    """
    获取每个视频起始的图片id
    通过这个id, 可以在 dataset 里面，找到当前视频的每个视频片段的起始图片id
    """
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx

def get_dict_start_idx_LFB(sequence_length, dataset):
    dataset_num_each = []
    dataset_num_each = dataset.get_num_each_video()
    start_idx_LFB = get_useful_start_idx(sequence_length, dataset_num_each)
    dict_index, dict_value = zip(*list(enumerate(start_idx_LFB)))
    dict_start_idx_LFB = dict(zip(dict_value, dict_index))

    return dict_start_idx_LFB


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