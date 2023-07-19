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
# parser.add_argument('--sample_rate', default=2, type=int, help='图片的采样率， fps原始为25')
# parser.add_argument('--predictor_model', default='resnet_lstm', help='特征提取的模型')
# parser.add_argument('--model_name', default='resnet_lstm', help='模型')
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

    train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
    train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
    test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
    test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

    device = cuda_test()
    if device == False:
        print("false")
        return
    
    # model = select_model()
    model_name = opt.model_name
    action = args.action
    if action == 'train':
        if model_name == "resnet50":
            import model.predictor.resnet50 as resnet50_model
            model = resnet50_model.resnet50()
            train.train(opt, model, train_dataset, test_dataset, device)

        if model_name == "resnet_lstm":

            import model.predictor.resnet_lstm as resnet_lstm_model
            import script.resnet_lstm as train
            model = resnet_lstm_model.resnet_lstm(opt)
            train.train(opt, model, train_dataset, test_dataset, device)

        if model_name == "tcn":
            """
            从tcn的参数来看，out_features = 7 应该是和阶段的类别数相关
            
            """
            out_features = 7
            mstcn_causal_conv = True
            mstcn_layers = 8
            mstcn_f_maps = 32
            mstcn_f_dim= 2048
            mstcn_stages = 2

            import model.predictor.tcn as tcn_model
            model = tcn_model.MultiStageModel(opt)
            import script.tcn as tcn_action

            tcn_action.train(opt, model, train_dataset, test_dataset, device)
            # (opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/resnet_lstm")

        if model_name == "TMR":

            ##
            import script.TMR as train
            train.train(opt, train_dataset, test_dataset, device)
            # (opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/resnet_lstm")

    if action == 'eval':
        """
        项目的验证和测试的代码，需要给出测试结果的可视化和每个视频的正确率
        ！！！！有一个小问题，就是，使用不同的 predictor 模型特征，训练好的 细化阶段 的模型，
        到时候需要一整个一起预测模型的训练结果，所以，需要整合成端到端的形式吗？
        （想尽可能一个脚本实现好）
        （重点是在线识别！！！）
        （在线使用 predictor 模型把
        """
        print("TODO")

    if args.action == 'extract':
        if model_name == "resnet50":
            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import model.predictor.resnet50 as resnet50
            import script.resnet50 as resnet50_model
            model = resnet50.resnet_feature()
            resnet50_model.extract(opt, model, train_dataset, test_dataset, device)
            # def extract(opt, model, train_dataset, test_dataset, device, save_dir = "./result/feature/resnet_lstm")    
        if model_name == "resnet_lstm":
            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import model.predictor.resnet_lstm as resnet_lstm
            import script.resnet_lstm as resnet_lstm_model
            model = resnet_lstm.resnet_lstm_feature(opt)
            resnet_lstm_model.extract(opt, model, train_dataset, test_dataset, device)

        if model_name == "tcn":
            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import model.predictor.tcn as tcn_model
            import script.tcn as tcn_action
            model = resnet_lstm_model.resnet_lstm(opt)
            tcn_action.extract(opt, model, train_dataset, test_dataset, device)


if __name__ == '__main__':  
    seed_everything()      
    run()
