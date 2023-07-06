import utils.opts as opts

import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import argparse
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection  import KFold

import dataset.cholec80 as dataset
from utils import *

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', help='需要执行的操作: train, test, extract')
parser.add_argument('--dataset', default="cholec80", help='选择的数据集')
parser.add_argument('--sample_rate', default=2, type=int, help='图片的采样率， fps原始为25')
# parser.add_argument('--predictor_model', default='resnet_lstm', help='特征提取的模型')
parser.add_argument('--model_name', default='resnet50', help='模型')
parser.add_argument('--model_path', default='', help='模型的路径，用来提取特征')
parser.add_argument('--device', type=str, default="cuda")

args, args2 = parser.parse_known_args()

print(args)
print(args2)

# 加载其他模型的参数（比如配置文件等等）
opt = opts.parse_opt()
# print(opt)

   
def seed_everything(seed=123): # original one is 123, 3407
    '''set seed for deterministic training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cuda_test():
    """
    如果GPU可用，就使用GPU0，如果没有空闲GPU，就啥也不干
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU可用")
        return device
    else:
        # device = torch.device("cpu")
        print("GPU不可用，也不想使用CPU，先去干点别的事情吧")
        return False

# from model.predictor import resnet_lstm
# model = resnet_lstm(opt)

def select_model():
    model_name = args.model
    if model_name == "resnet_lstm":
        from model.predictor import resnet_lstm
        model = resnet_lstm(opt)
        print(" resnet_lstm 加载成功 ！！！！")

    return model

def run():

    device = cuda_test()
    if device == False:
        print("false")
        return
    
    # model = select_model()
    model_name = args.model_name
    action = args.action
    if action == 'train':
        if model_name == "resnet50":
            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import script.resnet50 as train
            train.train(opt, train_dataset, test_dataset, device)
        if model_name == "resnet_lstm":
            print(" resnet_lstm 开始加载 ")
            import model.predictor.resnet_lstm as resnet_lstm_model
            model = resnet_lstm_model.resnet_lstm(opt)
            print(" resnet_lstm 加载成功 ！！！！")

            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import script.resnet_lstm as train
            train.train(opt, model, train_dataset, test_dataset, device)
            # (opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/resnet_lstm")
        if model_name == "TMR":

            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import script.TMR as train
            train.train(opt, train_dataset, test_dataset, device)
            # (opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/resnet_lstm")

    if action == 'eval':
        print("TODO")

    if args.action == 'extract':
        if model_name == "resnet_lstm":
            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import script.resnet_lstm as model
            model.extract(opt, train_dataset, test_dataset, device)
            # def extract(opt, model, train_dataset, test_dataset, device, save_dir = "./result/feature/resnet_lstm")    


if __name__ == '__main__':  
    seed_everything()      
    run()
