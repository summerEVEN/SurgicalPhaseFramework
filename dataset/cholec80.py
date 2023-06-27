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

class CholecDataset(Dataset):
    """
    
    """
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=default_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels[:,0]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase

    def __len__(self):
        return len(self.file_paths)


class FramewiseDataset(Dataset):
    """
    
    """
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
    

class VideoDataset(Dataset):
    """
    NETE 的 dataset 定义
    里面关于 mask 和 corss-validate 部分目前没有看明白
    """
    def __init__(self, dataset, root, sample_rate, video_feature_folder, blacklist=[]):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.blacklist = blacklist # for cross-validate
        self.videos = []
        self.labels = []
        self.hard_frames = []
        self.video_names = []
        if dataset =='cholec80':
            self.hard_frame_index = 7
        if dataset == 'm2cai16':
            self.hard_frame_index = 8 

        video_feature_folder = os.path.join(root, video_feature_folder)
        label_folder = os.path.join(root, 'annotation_folder')
        hard_frames_folder = os.path.join(root, 'hard_frames@2020')
        for v_f in os.listdir(video_feature_folder):
            if v_f.split('.')[0] in blacklist:
                continue
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
            v_label_file_abs_path = os.path.join(label_folder, v_f.split('.')[0] + '.txt')
            v_hard_frame_abs_path = os.path.join(hard_frames_folder, v_f.split('.')[0] + '.txt')
            labels = self.read_labels(v_label_file_abs_path)
            
            labels = labels[::sample_rate]
            videos = np.load(v_f_abs_path)[::sample_rate,]
            masks = self.read_hard_frames(v_hard_frame_abs_path,  self.hard_frame_index)
            masks = masks[::sample_rate]
            assert len(labels) == len(masks)

            self.videos.append(videos)
            self.labels.append(labels)
            self.hard_frames.append(masks)
            self.video_names.append(v_f)

        print('VideoDataset: Load dataset {} with {} videos.'.format(self.dataset, self.__len__()))


class TestVideoDataset(Dataset):
    '''
    SAHC 的 dataset 的定义
    读取 video_feature_folder 里面的视频特征，和对应的 labels 
    按照 sample_rate 采样

    一个视频对应一条数据吧(这样也就方便了后面计算每个视频的正确率)
    '''
    def __init__(self, dataset, root, sample_rate, video_feature_folder):
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.videos = []
        self.labels = []

        video_feature_folder = os.path.join(root, video_feature_folder)
        label_folder = os.path.join(root, 'annotation_folder')

        num_len = 0
        ans = 0

        for v_f in os.listdir(video_feature_folder):       
            v_f_abs_path = os.path.join(video_feature_folder, v_f)
            v_label_file_abs_path = os.path.join(label_folder, v_f.split('.')[0] + '.txt')
            labels = self.read_labels(v_label_file_abs_path) 
            labels = labels[::sample_rate]
            videos = np.load(v_f_abs_path)[::sample_rate,]
            num_len += videos.shape[0]
            self.videos.append(videos)
            self.labels.append(labels)
            phase = 1
            for i in range(len(labels)-1):
                    if labels[i] == labels[i+1]:
                        continue
                    else:
                        phase += 1
            ans += 1
            self.video_names.append(v_f)
       
        print('VideoDataset: Load dataset {} with {} videos.'.format(self.dataset, self.__len__()))

    def __len__(self):
        return len(self.videos)
  
    def __getitem__(self, item):
        video, label, video_name = self.videos[item], self.labels[item], self.video_names[item]
        return video, label, video_name
    
    def read_labels(self, label_file):
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()]
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels