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

# 定义 数据集的阶段 和 label 之间的字典
# 包含多个数据集的转换字典
# 实际上这个
phase2label_dicts = {
    'cholec80':{
    'Preparation':0,
    'CalotTriangleDissection':1,
    'ClippingCutting':2,
    'GallbladderDissection':3,
    'GallbladderPackaging':4,
    'CleaningCoagulation':5,
    'GallbladderRetraction':6},
    
    'm2cai16':{
    'TrocarPlacement':0,
    'Preparation':1,
    'CalotTriangleDissection':2,
    'ClippingCutting':3,
    'GallbladderDissection':4,
    'GallbladderPackaging':5,
    'CleaningCoagulation':6,
    'GallbladderRetraction':7}
    }

def phase2label(phases, phase2label_dict):
    """ 返回一段 phase 对应的 label；phases 里面包含多个 phase
    Args：
        phases：一段阶段的数据
        phase2label_dict：转换的字典
    
    Returns：
        labels：一段标签数据
    """
    labels = [phase2label_dict[phase] if phase in phase2label_dict.keys() else len(phase2label_dict) for phase in phases]
    return labels

def label2phase(labels, phase2label_dict):
    """ 根据 label，返回对应的 phase 字段
    Args：
        labels：一段标签数据
        phase2label_dict：转换的字典
    
    Returns：
        phases：一段数据
    """
    label2phase_dict = {phase2label_dict[k]:k for k in phase2label_dict.keys()}
    phases = [label2phase_dict[label] if label in label2phase_dict.keys() else 'HardFrame' for label in labels]
    return phases

class FramewiseDataset(Dataset):
    """
    __init__ 函数：
    根据路径读取 数据集的label 和图片信息，以及图片的路径
    
    """
    def __init__(self, dataset, root, down_sampling = 25, label_folder='phase_annotations', video_folder='cutMargin', blacklist=[]):
        """ 逐帧的数据集 的初始化方法
        实现功能：
            根据文件夹和路径，按照采样率，得到视频的数据集

        Args：
            dataset：数据集的名称（cholec80，m2cai16等）
            root：数据集的根目录
            down_sampling：图片的采样率（默认值25）
            label_folder：标签数据的文件夹名 （root/phase_annotations/video01-phase.txt)
            video_folder：视频图片数据的文件夹名 (root/cutMargin/1/1.png)
            blacklist：读取数据集的时候，跳过 balcklist 里面的文件夹

        Returns：
            无
        
        """
        self.dataset = dataset
        self.blacklist= blacklist
        self.imgs = []
        self.labels = []
        self.num_each_video = []
        # self.imgs： 图片路径集合
        # self.labels： 图片标签集合
        # self.num_each_video： 每个视频包含的图片的数量 （方便后期每个视频的可视化操作）

        label_folder = os.path.join(root, label_folder)
        video_folder = os.path.join(root, video_folder)
        video_folders = os.listdir(video_folder)
        video_folders.sort()
        # 排个序（忘记是什么动机了hhh，就按照视频的名称排个序）

        for v in video_folders:
            if v in blacklist:
                continue
            v_abs_path = os.path.join(video_folder, v)
            v_label_file_abs_path = os.path.join(label_folder, "video" + v + '-phase.txt')
            labels = self.read_labels(v_label_file_abs_path)
            # images = os.listdir(v_abs_path)

            # assert len(labels) == len(images)
            
            # for image in images:
            #     image_index = int(image.split('.')[0])
            #     self.imgs.append(os.path.join(v_abs_path, image))
            #     self.labels.append(labels[image_index])
            for i in range(len(labels)):
                if i * down_sampling >= len(labels):
                    self.num_each_video.append(i)
                    break
                self.labels.append(labels[i * down_sampling])
                # print(i * down_sampling, len(labels), v)
                self.imgs.append(os.path.join(v_abs_path, str(i * down_sampling) + ".jpg")) 

                # print(i)
                # print(os.path.join(v_abs_path, str(i) + ".jpg"))
                # print("----------------------")
                # print()

        self.transform = self.get_transform("opt")

        print('FramewiseDataset: Load dataset {} with {} images.'.format(self.dataset, self.__len__()))

    def __len__(self):
        """
        返回数据集的大小（图片数）
        """
        return len(self.imgs)

    def __getitem__(self, id):
        """ 根据 id 获得数据集里面对应图片的信息
        Args：
            id：图片的序号
        
        Returns：
            img：经过处理后的图片tensor类型数据
            label：图片对应的标签（int）
            id：图片的序号
            img_path: 图片的保存路径
        """
        img, label, img_path = self.transform(default_loader(self.imgs[id])), self.labels[id], self.imgs[id]
        return img, label, id, img_path

    def get_transform(self, opt):
        """ 图片预处理的方法
        
        根据传递的 opt 的参数，选择图片的预处理的方法
        这个预处理打算借鉴一下 TMR 的方法，设计一下预处理相关的一些参数

        不同的模型的图片的size可能不一样
        这个的话，后续看看如何统一管理
        抽取出来，写成一个单独的模块？？？
        """
        # return transforms.Compose([
        #         transforms.Resize((299,299)),
        #         transforms.ToTensor()
        # ])
        # TODO
    
        return transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor()
        ])

    def read_labels(self, label_file):
        """ 把 label.txt 里面的 phase 都转变成 label
        Args:
            label_file: 读取的文件名（label.txt)
        
        Return:
            labels: 就是 labels 哈哈哈
        """
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()]
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels

    def get_num_each_video(self):
        """
        返回一个list, 里面包含每个视频的图片数。
        """
        return self.num_each_video

class VideoDataset(Dataset):
    """
    NETE 的 dataset 定义
    里面关于 mask 和 corss-validate 部分目前没有看明白
    """
    # TODO 还没有整合
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

    # TODO 还没有整合

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
  
    def __getitem__(self, id):
        video, label, video_name = self.videos[id], self.labels[id], self.video_names[id]
        return video, label, video_name
    
    def read_labels(self, label_file):
        with open(label_file,'r') as f:
            phases = [line.strip().split('\t')[1] for line in f.readlines()]
            labels = phase2label(phases, phase2label_dicts[self.dataset])
        return labels
    
if __name__ == "__main__":
    """
    UNIT TEST
    """
    current_path = os.getcwd()
    print("当前程序运行的路径：", current_path)
    label_folder = 'phase_annotations'
    video_folder = 'cutMargin'

    path = os.path.join(current_path, "../Dataset/cholec80")
    if os.path.exists(path):
        traindataset = FramewiseDataset("cholec80", path, label_folder, video_folder)
        print(len(traindataset))
        
    print()
    print("!!!")