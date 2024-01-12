import torch
import numpy as np
from torch import nn
from typing import Union
import numpy.typing as npt

class DoubleConv(nn.Module):
    """
    DoubleConv: a wrapper class for a block of 3x3 convolutions, BatchNorms, and ReLUs
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Union[int, None]=None) -> None:
        """
        __init__: capture number of channels for the input and output

        Inputs:
            in_channels (int): the number of channels for the input
            out_channels (int): the number of channels for the output
            mid_channels (Union[int, None]; default=None): the number of channels for the ouput of the first conv. defaults to None (i.e. same as out_channels)
        """
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward: the forward pass, which passes the input through all layers in the block

        Inputs:
            x (torch.Tensor): the input, with shape (N, C_in, H, W).
        Outputs:
            *unnamed* (torch.Tensor): the input post-convolution, with shape (N, C_out, H_out, W_out)
        """
        return self.conv(x)

class DownSample(nn.Module):
    """
    DownSample: a block of DoubleConv blocks and MaxPools, given a specific depth
    """

    def __init__(self, depth: int, in_channels: int) -> None:
        """
        __init__: capture the block's depth and the number of input channels

        Inputs:
            depth (int): the number of DoubleConv and MaxPool cycles to undergo
            in_channels (int): the number of channels for the input
        """
        super().__init__()
        self.layers = nn.Sequential(
            DoubleConv(in_channels, 4),
            nn.MaxPool2d(2)
        )
        for layer in range(1, depth):
            self.layers.append(DoubleConv(2 ** (1 + layer), 2 ** (2 + layer)))
            self.layers.append(nn.MaxPool2d(2))

class UpSample(nn.Module):
    """
    UpSample: DoubleConv blocks and upsampling modules, given a specific depth. capped with a 1x1 convolution
    """
    def __init__(self, depth: int, out_channels: int) -> None:
        """
        __init__: capture the block's depth and the number of output channels
        
        Inputs:
            depth (int): the number of DoubleConv and upsampling cycles to undergo
            out_channels (int): the number of channels for the output
        """
        super().__init__()
        self.layers = nn.Sequential()
        for layer in range(depth - 1, 0, -1):
            self.layers.append(nn.ConvTranspose2d(2 ** (3 + layer), 2 ** (2 + layer), 2, stride=2))
            self.layers.append(DoubleConv(2 ** (3 + layer), 2 ** (2 + layer)))
        self.layers.append(nn.ConvTranspose2d(8, 4, 2, stride=2))
        self.layers.append(DoubleConv(8, 4))
        self.layers.append(nn.Conv2d(4, out_channels, kernel_size=1))

class UNet(nn.Module):
    """
    UNet: the UNet architecture
    """

    def __init__(self, in_channels: int=1, out_channels: int=1, depth: int=5):
        """
        __init__: capture the depth of the UNet and the number of channels for the input and output

        Inputs:
            in_channels (int): the number of input channels
            out_channels (int): the number of output channels
            depth (int): the number of downsample/upsample cycles
        """
        super().__init__()
        self.down = DownSample(depth, in_channels)
        self.bridge = DoubleConv(2 ** (1 + depth), 2 ** (2 + depth))
        self.up = UpSample(depth, out_channels)
            
    def forward(self, x, sigmoid=False):
        """
        forward: the forward pass, which passes the input through all layers in the model

        Inputs:
            x (torch.Tensor): the input sample
            sigmoid (bool; default=Fase): a boolean of whether to return the logits or the post-sigmoid output.
        """
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

class Hourglass(nn.Module):
    """
    Hourglass: a number of stacked UNets. heavily based on the CenterNet paper
    """

    def __init__(self, in_channels=1, out_channels=1, depth=2):
        """
        __init__: capture the depth of the entire Hourglass network and the number of channels for the input and output

        Inputs:
            in_channels (int; default=1): the number of input channels
            out_channels (int; default=1): the number of output channels
            depth (int; default=1): the number of UNets to use
        """
        super().__init__()
        self.layers = nn.Sequential()
        if (depth == 1):
            self.layers.append(UNet(in_channels, out_channels))
        else:
            self.layers.append(UNet(in_channels, 4))
            for i in range(1, depth - 1):
                self.layers.append(UNet(4, 4))
            self.layers.append(UNet(4, out_channels))
            
    def forward(self, x, sigmoid=True):
        """
        forward: the forward pass, which passes the input through all layers in the model

        Inputs:
            x (torch.Tensor): the input sample
            sigmoid (bool; default=Fase): a boolean of whether to return the logits or the post-sigmoid output.
        """
        x = self.layers(x)
        if sigmoid:
            return torch.sigmoid(x)
        return x