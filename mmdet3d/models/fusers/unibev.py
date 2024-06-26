import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["UniFuser"]


@FUSERS.register_module()
class UniFuser(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.input_shape = (180, 180)
        
        self.conv3x3 = nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.avg = nn.AdaptiveAvgPool2d(self.input_shape)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()



    def forward(self, inputs):
        concat = torch.cat(inputs, dim=1)
        
        x = self.conv3x3(concat)
        x = self.bnorm(x)
        x = self.relu(x)

        x1 = self.avg(x)
        x1 = self.conv1x1(x1)
        x1 = self.sigmoid(x1)

        out = x * x1
        
        return out
