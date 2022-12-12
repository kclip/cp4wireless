import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class VGG16(nn.Module):
    def __init__(self, num_classes=24):
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.stride = 1
        self.padding = 1
        self.fc1 = nn.Linear(512, 128)  # fully connected (FC) layer
        self.fc2 = nn.Linear(128, 128) # fully connected (FC) layer
        self.fc3 = nn.Linear(128, num_classes) # fully connected (FC) layer
        self.softmax = nn.Softmax(dim=1)
        self.activ = nn.SELU()
    def forward(self, x, var, if_softmax_out=False):
        #print(f'VGG16.forward x={x} var={var}')
        if var is None:
            var = list(map(lambda p: p[0], zip(self.parameters())))
        else:
            pass
        idx = 0
        num_fc_layers = 3 
        while idx < len(var):
            gap = 2 # weight and bias
            if idx > 0:
                x = self.activ(x)
            else:
                pass
            if idx < len(var)-2*num_fc_layers:  # len(var) = 14+6, 20 - 6 = 14, idx: 0,1,...,13: CNN
                cnn_w, cnn_b = var[idx], var[idx + 1]
                x = F.conv1d(input = x, weight = cnn_w, bias = cnn_b, stride=self.stride, padding=self.padding)
                x = F.max_pool1d(x, kernel_size=2, stride=2)
            else: # 14,..., 19
                fc_w, fc_b = var[idx], var[idx + 1]  # weight and bias for fc
                x = torch.reshape(x, (x.shape[0], -1))
                x = F.linear(input = x, weight = fc_w, bias = fc_b)
            idx += gap
        if if_softmax_out:
            x = self.softmax(x)
        else:
            pass
        return x

class VGG_small(nn.Module):
    def __init__(self, num_classes=24):
        super(VGG_small, self).__init__()
        self.fc1 = nn.Linear(2*1024, 4)  # fully connected (FC) layer
        self.fc2 = nn.Linear(4, num_classes) # fully connected (FC) layer
        self.softmax = nn.Softmax(dim=1)
        self.activ = nn.SELU()
    def forward(self, x, var, if_softmax_out=False):
        if var is None:
            var = list(map(lambda p: p[0], zip(self.parameters())))
        else:
            pass
        x = x.reshape(x.shape[0],-1)
        x = F.linear(input = x, weight = var[0], bias = var[1])
        x = self.activ(x)
        x = F.linear(input = x, weight = var[2], bias = var[3])
        if if_softmax_out:
            x = self.softmax(x)
        return x