"""
models: a number of UNet-based models and helper functions
"""

import numpy as np
import torch
from torch import nn

""" #########################
DoubleConv: a wrapper class for a block of 3x3 convolutions, BatchNorms, and ReLUs

__init__: capture number of channels for the input and output
Args:
    in_channels: the number of channels for the input
    out_channels: the number of channels for the output
    mid_channels: the number of channels for the ouput of the first conv. defaults to None (i.e.
        same as out_channels)

forward: the forward pass, which passes the input through all layers in the block
Args:
    x: the input, with shape (N, C_in, H, W).
Returns:
    *unnamed*: the input post-convolution, with shape (N, C_out, H_out, W_out)
######################### """
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if (mid_channels == None):
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
""" ######################### """

""" #########################
DownSample: a block of DoubleConv blocks and MaxPools, given a specific depth

__init__: capture the block's depth and the number of input channels
Args:
    depth: the number of DoubleConv and MaxPool cycles to undergo
    in_channels: the number of channels for the input
    dropout: the probability of dropout
######################### """
class DownSample(nn.Module):
    def __init__(self, depth, in_channels, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            DoubleConv(in_channels, 4),
            #nn.Dropout(p=dropout),
            nn.MaxPool2d(2)
        )
        for layer in range(1, depth):
            self.layers.append(DoubleConv(2 ** (1 + layer), 2 ** (2 + layer)))
            self.layers.append(nn.MaxPool2d(2))
""" ######################### """

""" #########################
UpSample: DoubleConv blocks and upsampling modules, given a specific depth. capped with a 1x1
    convolution

__init__: capture the block's depth and the number of output channels
Args:
    depth: the number of DoubleConv and upsampling cycles to undergo
    out_channels: the number of channels for the output
    dropout: the probability of dropout
######################### """
class UpSample(nn.Module):
    def __init__(self, depth, out_channels, dropout):
        super().__init__()
        self.layers = nn.Sequential()
        for layer in range(depth - 1, 0, -1):
            self.layers.append(nn.ConvTranspose2d(2 ** (3 + layer), 2 ** (2 + layer),
                                                  2, stride=2))
            self.layers.append(DoubleConv(2 ** (3 + layer), 2 ** (2 + layer)))
            #self.layers.append(nn.Dropout(p=dropout))
        self.layers.append(nn.ConvTranspose2d(8, 4, 2, stride=2))
        self.layers.append(DoubleConv(8, 4))
        self.layers.append(nn.Conv2d(4, out_channels, kernel_size=1))
""" ######################### """

""" #########################
UNet: the UNet architecture

__init__: capture the depth of the UNet and the number of channels for the input and output
Args:
    in_channels: the number of input channels
    out_channels: the number of output channels
    depth: the number of downsample/upsample cycles
    dropout: the probability of dropout. Default: 0 (no dropout)

forward: the forward pass, which passes the input through all layers in the model
Args:
    x: the input sample
    sigmoid: a boolean of whether to return the logits or the post-sigmoid output.
        Default: False
######################### """
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=5, dropout=0):
        super().__init__()
        self.down = DownSample(depth, in_channels, dropout)
        self.bridge = DoubleConv(2 ** (1 + depth), 2 ** (2 + depth))
        self.up = UpSample(depth, out_channels, dropout)
            
    def forward(self, x, sigmoid=False):
        x_trace = [x]
        out_trace = []
        for module in self.down.layers.children():
            x_trace.append(module(x_trace[-1]))
            if (type(module) == DoubleConv):
                out_trace.append(x_trace[-1])
        x_trace.append(self.bridge(x_trace[-1]))
        for module in self.up.layers.children():
            if (type(module) == DoubleConv):
                head = out_trace.pop()
                tail = x_trace[-1]
                pad = (int(np.floor((head.shape[3] - tail.shape[3]) / 2)),
                       int(np.ceil((head.shape[3] - tail.shape[3]) / 2)),
                       int(np.floor((head.shape[2] - tail.shape[2]) / 2)),
                       int(np.ceil((head.shape[2] - tail.shape[2]) / 2)))
                tail = nn.functional.pad(tail, pad)
                x_trace[-1] = torch.cat([head, tail], axis=1)
            x_trace.append(module(x_trace[-1]))
        if (sigmoid):
            return torch.sigmoid(x_trace[-1])
        return x_trace[-1]
""" ######################### """

""" #########################
Hourglass: a number of stacked UNets. heavily based on the CenterNet paper

__init__: capture the depth of the entire Hourglass network and the number of channels for
    the input and output
Args:
    in_channels: the number of input channels. Default: 1
    out_channels: the number of output channels. Default: 1
    depth: the number of UNets to use. Default: 2
    dropout: the probability of dropout. Default: 0 (no dropout)

forward: the forward pass, which passes the input through all layers in the model
Args:
    x: the input sample
    sigmoid: a boolean of whether to return the logits or the post-sigmoid output.
        Default: True
######################### """
class Hourglass(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, depth=2, dropout=0):
        super().__init__()
        self.layers = nn.Sequential()
        if (depth == 1):
            self.layers.append(UNet(in_channels, out_channels, dropout=dropout))
        else:
            self.layers.append(UNet(in_channels, 4, dropout=dropout))
            for i in range(1, depth - 1):
                self.layers.append(UNet(4, 4))
            self.layers.append(UNet(4, out_channels, dropout=dropout))
            
    def forward(self, x, sigmoid=True):
        x = self.layers(x)
        if sigmoid:
            return torch.sigmoid(x)
        return x
""" ######################### """