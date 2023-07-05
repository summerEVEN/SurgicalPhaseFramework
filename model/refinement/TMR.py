import torch
import torch.nn as nn
from torchvision import models
import torch.nn.init as init
import torch.nn.functional as F

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
        self.fc_c = nn.Linear(512, 7)
        self.fc_h_c = nn.Linear(1024, 512)
        self.nl_block = NLBlock()
        self.dropout = nn.Dropout(p=0.5)
        
        self.time_conv = TimeConv()

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc_c.weight)
        init.xavier_uniform_(self.fc_h_c.weight)

        self.sequence_length = opt.sequence_length
        self.is_time_conv = opt.is_time_conv

    def forward(self, x, long_feature):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, self.sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = y[self.sequence_length - 1::self.sequence_length]

        if self.is_time_conv:
            Lt = self.time_conv(long_feature)
            y_1 = self.nl_block(y, Lt)
        else: 
            y_1 = self.nl_block(y, long_feature)
        y = torch.cat([y, y_1], dim=1)
        y = self.dropout(self.fc_h_c(y))
        y = F.relu(y)
        y = self.fc_c(y)

        # y = y[self.sequence_length - 1::self.sequence_length]
        return y
    

class NLBlock(nn.Module):
    def __init__(self, feature_num=512):
        super(NLBlock, self).__init__()
        self.linear1 = nn.Linear(feature_num, feature_num)
        self.linear2 = nn.Linear(feature_num, feature_num)
        self.linear3 = nn.Linear(feature_num, feature_num)
        self.linear4 = nn.Linear(feature_num, feature_num)
        self.layer_norm = nn.LayerNorm([1, 512])
        self.dropout = nn.Dropout(0.2)

        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)
        init.xavier_uniform_(self.linear4.weight)

    def forward(self, St, Lt):
        St_1 = St.view(-1, 1, 512)
        St_1 = self.linear1(St_1)
        Lt_1 = self.linear2(Lt)
        Lt_1 = Lt_1.transpose(1, 2)
        SL = torch.matmul(St_1, Lt_1)       
        SL = SL * ((1/512)**0.5)
        SL = F.softmax(SL, dim=2)
        Lt_2 = self.linear3(Lt)
        SLL = torch.matmul(SL, Lt_2)
        SLL = self.layer_norm(SLL)
        SLL = F.relu(SLL)
        SLL = self.linear4(SLL)
        SLL = self.dropout(SLL)
        SLL = SLL.view(-1, 512)
        return (St+SLL)

    
class TimeConv(nn.Module):
    def __init__(self):
        super(TimeConv, self).__init__()
        self.timeconv1 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.timeconv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.timeconv3 = nn.Conv1d(512, 512, kernel_size=7, padding=3)
        self.maxpool_m = nn.MaxPool1d(2, stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d((512,1))

    def forward(self, x):
        x = x.transpose(1, 2)
        
        x1 = self.timeconv1(x)
        y1 = x1.transpose(1, 2)
        y1 = y1.view(-1,30,512,1)

        x2 = self.timeconv2(x)
        y2 = x2.transpose(1, 2)
        y2 = y2.view(-1,30,512,1)

        x3 = self.timeconv3(x)
        y3 = x3.transpose(1, 2)
        y3 = y3.view(-1,30,512,1)

        x4 = F.pad(x, (1,0), mode='constant', value=0)
        x4 = self.maxpool_m(x4)
        y4 = x4.transpose(1, 2)
        y4 = y4.view(-1,30,512,1)

        y0 = x.transpose(1, 2)
        y0 = y0.view(-1,30,512,1)

        y = torch.cat((y0,y1,y2,y3,y4), dim=3)
        y = self.maxpool(y)
        y = y.view(-1,30,512)
        
        return y

