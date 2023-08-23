import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BaseCausalTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        print(num_layers)
        super(BaseCausalTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualCausalLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.channel_dropout = nn.Dropout2d()
        # self.downsample = nn.Linear(num_f_maps,num_f_maps, kernel_size=3, stride=2,dilation=3)
        # self.center = torch.nn.Parameter(torch.zeros(1, 64, num_classes), requires_grad=False)
        self.num_classes = num_classes
        
    def forward(self, x, labels=None, mask=None, test=False):
        # x = x.permute(0,2,1) # (bs,l,c) -> (bs, c, l)
        
        if mask is not None:
            # print(x.size(),mask.size())
            x = x * mask
        
        x= x.unsqueeze(3) # of shape (bs, c, l, 1)
        x = self.channel_dropout(x)
        x = x.squeeze(3)
        
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        x = self.conv_out(out) # (bs, c, l)
        return out, x  
    
class DilatedResidualCausalLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, padding=None):
        super(DilatedResidualCausalLayer, self).__init__()
        if padding == None:
            self.padding = 2 * dilation
        else:
            self.padding=padding
        # causal: add padding to the front of the input
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation) #
        # self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.pad(x, [self.padding, 0], 'constant', 0) # add padding to the front of input
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)