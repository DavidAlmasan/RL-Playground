import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kern_sz, stride, use_bn=False, residual=False):
        super(ConvBlock, self).__init__()

        self.block = [nn.Conv2d(in_ch, out_ch, kern_sz, stride),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(inplace=True)] if use_bn else \
                    \
                     [nn.Conv2d(in_ch, out_ch, kern_sz, stride),
                      nn.ReLU(inplace=True)]     
        
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        x = self.block(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self, len_action_space):
        super(SmallCNN, self).__init__()

        self.conv1 = ConvBlock(4, 16, 8, 4)
        self.conv2 = ConvBlock(16, 32, 4, 2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(9 * 9 * 32, 256)
        self.fc2 = nn.Linear(256, len_action_space)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1)
        return x

        