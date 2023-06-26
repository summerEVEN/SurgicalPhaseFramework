"""
目前的几个项目的 dataset 的设计不一样，考虑吧 dataset 统一一下

TMR网络，他使用的比较难懂不直接的方法，可能只是为了得到连续帧的序列
但是这样的操作非常复杂且难懂

NETE里面有一个 FramewiseDataset 也许可以利用起来
FramewiseDataset 里面的数据可以按视频帧的顺序保存起来
然后每次从里面挑选合适的起始帧，输入进行训练，这样还是有一定的可能性实现的
"""

"""

❗❗❗❗❗❗❗❗❗❗❗❗
这个 dataset 的设计，牵扯到最后数据集的训练集和测试集的文件夹的摆放
首先的想法是按照 NETE 的设计来，训练集和测试集分文件夹保存

"""


from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import os
import numpy as np


class FramewiseDataset(Dataset):
    def __init__(self, dataset, root, label_folder='annotation_folder', video_folder='image_folder', blacklist=[]):
        self.dataset = dataset
        self.blacklist= blacklist
        self.imgs = []
        self.labels = []

        label_folder = os.path.join(root, label_folder)
        video_folder = os.path.join(root, video_folder)
        for v in os.listdir(video_folder):
            if v in blacklist:
                continue
            v_abs_path = os.path.join(video_folder, v)
            v_label_file_abs_path = os.path.join(label_folder, v + '.txt')
            labels = self.read_labels(v_label_file_abs_path)
            images = os.listdir(v_abs_path)

            assert len(labels) == len(images)
            for image in images:
                image_index = int(image.split('.')[0])
                self.imgs.append(os.path.join(v_abs_path, image))
                self.labels.append(labels[image_index])
        self.transform = self.get_transform()

        print('FramewiseDataset: Load dataset {} with {} images.'.format(self.dataset, self.__len__()))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img, label, img_path = self.transform(default_loader(self.imgs[item])), self.labels[item], self.imgs[item]
        return img, label, img_path

    def get_transform(self):
        return transforms.Compose([
                transforms.Resize((299,299)),
                transforms.ToTensor()
        ])

    def read_labels(self, label_file):
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()]
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels