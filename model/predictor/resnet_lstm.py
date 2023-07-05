"""
定义一个 resnet_lstm 提取网络的 空间-时间特征

发现一个需要修改的点：
原来的 TMR 网络的代码里面，有很多地方会使用全局参数
目前打算的修改方法：
1. 修改函数的参数，简单粗暴地把所有参数 opt 传进来
2. 按需传递参数

两种方法都可以，具体看个人习惯吧

"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.init as init


class resnet_lstm(torch.nn.Module):
    def __init__(self, opt):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, 7)
        self.dropout = nn.Dropout(p=0.2)
        self.sequence_length = opt.sequence_length

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, self.sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.dropout(y)
        y = self.fc(y)

        y = y[self.sequence_length - 1::self.sequence_length]
        return y
    
class resnet_lstm_feature(torch.nn.Module):
    def __init__(self, opt):
        super(resnet_lstm_feature, self).__init__()
        resnet = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)

        # self.fc = nn.Linear(512, 7)
        # self.dropout = nn.Dropout(p=0.2)

        self.sequence_length = opt.sequence_length

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        # init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, self.sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)

        # y = self.dropout(y)
        # y = self.fc(y)

        y = y[self.sequence_length - 1::self.sequence_length]
        return y