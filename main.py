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
# parser.add_argument('--model_path', default='', help='模型的路径，用来提取特征')
parser.add_argument('--device', type=str, default="cuda")

args, args2 = parser.parse_known_args()
# parse_known_args() 只更新已知的参数，多余的参数传递给后面的 parser 使用

print(args)
print(args2)

# 加载其他模型的参数（比如配置文件里面的参数等等）
opt = opts.parse_opt()

   
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
    如果GPU可用，就使用GPU，如果没有空闲GPU，就啥也不干
    还有一些问题，需要判断GPU是否被占用（使用 is_avaliable()　函数判断不可行）
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        print("GPU可用")
        return device
    else:
        # device = torch.device("cpu")
        print("GPU不可用，也不想使用CPU，先去干点别的事情吧")
        return False

def run():

    # 这里还有点问题，cuda_test函数存在点问题
    device = cuda_test()
    if device == False:
        print("false")
        return

    model_name = opt.model_name
    action = args.action
    if action == 'train':
        if model_name == "resnet50":
            # 加载 测试的数据集 和 训练的数据集
            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import model.predictor.resnet50 as resnet50_model
            import script.resnet50 as train
            model = resnet50_model.resnet50()
            train.train(opt, model, train_dataset, test_dataset, device)

        if model_name == "resnet_lstm":
            # 加载 测试的数据集 和 训练的数据集
            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import model.predictor.resnet_lstm as resnet_lstm_model
            import script.resnet_lstm as train
            model = resnet_lstm_model.resnet_lstm(opt)
            train.train(opt, model, train_dataset, test_dataset, device)

        if model_name == "tcn":
            """
            从tcn的参数来看，out_features = 7 应该是和阶段的类别数相关
            
            """
            # 加载 测试的数据集 和 训练的数据集
            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import model.predictor.tcn as tcn_model
            model = tcn_model.MultiStageModel(opt)
            import script.tcn as tcn_action

            tcn_action.train(opt, model, train_dataset, test_dataset, device)
            # (opt, model, train_dataset, test_dataset, device, save_dir = "./result/model/resnet_lstm")

        if model_name == "TMR":
            # 加载 测试的数据集 和 训练的数据集
            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import script.TMR as train
            import model.refinement.TMR as TMR_model
            model = TMR_model.resnet_lstm(opt)
            train.train(opt, model, train_dataset, test_dataset, device)
        
        if model_name == "SAHC":
            # from prototype import hierarch_train,base_predict
            # from utils import *
            # from hierarch_tcn2 import Hierarch_TCN2
            from script.SAHC import hierarch_train
            from model.refinement.SAHC.hierarch_tcn2 import Hierarch_TCN2

            num_stages = 3  # refinement stages
            # if args.dataset == 'm2cai16':
            #     num_stages = 2 # for over-fitting
            num_layers = 12 # layers of prediction tcn e
            num_f_maps = 64
            dim = 2048
            sample_rate = opt.sample_rate
            test_sample_rate = opt.test_sample_rate
            # num_classes = len(phase2label_dicts[args.dataset])
            num_classes = 7
            opt.num_classes = num_classes
            # print(opt.num_classes)
            num_layers_PG = opt.num_layers_PG
            num_layers_R = opt.num_layers_R
            num_R = opt.num_R

            video_traindataset = dataset.TestVideoDataset(opt.dataset, opt.dataset_path.format(opt.dataset) + '/train_dataset', sample_rate, 'video_feature')
            video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=False, drop_last=False)
            video_testdataset = dataset.TestVideoDataset(opt.dataset, opt.dataset_path.format(opt.dataset) + '/test_dataset', test_sample_rate, 'video_feature')
            video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False) 

            base_model=Hierarch_TCN2(opt, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
            # base_model=Hierarch_TCN2(opt)
            model_save_dir = 'result/model/{}/'.format(args.dataset)
            print("device", device)
            hierarch_train(opt, base_model, video_train_dataloader, video_test_dataloader, device, save_dir=model_save_dir, debug=True)

    if action == 'eval':
        """
        项目的验证和测试的代码，需要给出测试结果的可视化和每个视频的正确率
        ！！！！有一个小问题，就是，使用不同的 predictor 模型特征，训练好的 细化阶段 的模型，
        到时候需要一整个一起预测模型的训练结果，所以，需要整合成端到端的形式吗？
        （想尽可能一个脚本实现好）
        （重点是在线识别！！！）
        （在线使用 predictor 模型把
        """
        if model_name == "SAHC":
            from script.SAHC import evaluate_and_visualize
            from model.refinement.SAHC.hierarch_tcn2 import Hierarch_TCN2

            num_stages = 3  # refinement stages
            # if args.dataset == 'm2cai16':
            #     num_stages = 2 # for over-fitting
            num_layers = 12 # layers of prediction tcn e
            num_f_maps = 64
            dim = 2048
            sample_rate = opt.sample_rate
            test_sample_rate = opt.test_sample_rate
            # num_classes = len(phase2label_dicts[args.dataset])
            num_classes = 7
            opt.num_classes = num_classes
            # print(opt.num_classes)
            num_layers_PG = opt.num_layers_PG
            num_layers_R = opt.num_layers_R
            num_R = opt.num_R

            video_traindataset = dataset.TestVideoDataset(opt.dataset, opt.dataset_path.format(opt.dataset) + '/train_dataset', sample_rate, 'video_feature')
            video_train_dataloader = DataLoader(video_traindataset, batch_size=1, shuffle=False, drop_last=False)
            video_testdataset = dataset.TestVideoDataset(opt.dataset, opt.dataset_path.format(opt.dataset) + '/test_dataset', test_sample_rate, 'video_feature')
            video_test_dataloader = DataLoader(video_testdataset, batch_size=1, shuffle=False, drop_last=False) 

            base_model=Hierarch_TCN2(opt, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
            # base_model=Hierarch_TCN2(opt)
            # model_save_dir = 'result/model/{}/'.format(args.dataset)
            print("device", device)

            evaluate_and_visualize(opt, base_model, video_test_dataloader, device)


        if model_name == "TMR":
            # 加载 测试的数据集 和 训练的数据集
            train_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "train_dataset")
            train_dataset = dataset.FramewiseDataset(args.dataset, train_path)
            test_path = os.path.join(os.getcwd(), "../../Dataset/{}".format(args.dataset), "test_dataset")
            test_dataset = dataset.FramewiseDataset(args.dataset, test_path)

            import script.TMR as TMR
            import model.refinement.TMR as TMR_model
            model = TMR_model.resnet_lstm(opt)
            # TMR.evaluate_and_visualize(opt, model, test_dataset, device)
            TMR.evaluate_and_visualize(opt, model, train_dataset, device)

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
            # tcn 不需要预先提取特征
            import model.predictor.tcn as tcn_model
            import script.tcn as tcn_action
            model = resnet_lstm_model.resnet_lstm(opt)
            tcn_action.extract()


if __name__ == '__main__':  
    seed_everything()      
    run()
