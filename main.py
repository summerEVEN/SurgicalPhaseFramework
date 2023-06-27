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

from dataset import *
from utils import *
import action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', help='需要执行的操作: train, eval, ectract')
parser.add_argument('--dataset', default="cholec80", help='选择的数据集')
parser.add_argument('--sample_rate', default=2, type=int, help='图片的采样率， fps原始为25')
parser.add_argument('--predictor_model', default='resnet_lstm', help='特征提取的模型')
parser.add_argument('--refine_model', default='gru', help='细化阶段的模型')
parser.add_argument('--model_path', default='', help='模型的路径，用来提取特征')
args = parser.parse_args()
   
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
   
opt = opts.parse_opt()
from model.predictor import resnet_lstm
model = resnet_lstm(opt)

if __name__ == '__main__':
    if args.action == 'train':
        action.train(model)

    if args.action == 'eval':
        action.test(model)

    if args.action == 'extract':
        action.extract(model)
