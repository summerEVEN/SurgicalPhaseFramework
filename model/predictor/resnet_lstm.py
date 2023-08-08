import torch
import torch.nn as nn
from torchvision import models
import torch.nn.init as init


class resnet_lstm(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            opt: opt 参数包含配置文件和配置文件脚本（./utils/opts.py）里面的所有参数            
        """
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
    """
    和上面的网络的不同之处，就是注释掉的那部分
    上面的是用于训练的网络结构，最后的输出是预测的label

    这个是用于提取特征，最后的输出是大小为 512 的特征 
    （不需要训练，直接加载使用上面的网络结构训练好的参数即可）
    （关于 512 这个数值，这里是把它设置为固定值，但是其他的特征提取网络输出的特征大小不一定是 512）
    （可能也需要添加一个参数加入到 opt.py 里？？）
    """
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