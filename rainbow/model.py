import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kern_sz, use_bn=False, residual=False):
        super(ConvBlock, self).__init__()

        self.block = [nn.Conv2d(in_ch, out_ch, kern_sz),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU(inplace=True)] if use_bn else \
                    \
                     [nn.Conv2d(in_ch, out_ch, kern_sz),
                      nn.ReLU(inplace=True)]     
        
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        x = self.block(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.conv1 = ConvBlock(4, 16, 8)
        self.conv2 = ConvBlock(16, 32, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

        