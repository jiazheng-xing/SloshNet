import torch.nn as nn
import numpy  as np
import torch.nn.functional as F
import  torch
class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Conv3x3(nn.Module):
    def __init__(self, C_in, C_out, stride, affine= True):
        super().__init__()
        self.conv2 = self.conv3x3(C_in, C_out, stride)
        self.bn2 = nn.BatchNorm2d(C_in, affine=affine)
        self.relu = nn.ReLU()

    def conv3x3(slef, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)
    
    def forward(self, x):
        out = self.relu(x)
        out = self.bn2(out)
        out = self.conv2(out)

        return  out

class SpatialModulation(nn.Module):
    def __init__(
            self,
            inplanes=[1024, 2048],
            planes=2048,
    ):
        super(SpatialModulation, self).__init__()

        self.spatial_modulation = nn.ModuleList()
        for i, dim in enumerate(inplanes):
            op = nn.ModuleList()
            ds_factor = planes // dim
            ds_num = int(np.log2(ds_factor))
            if ds_num < 1:
                op = Identity()
            else:
                for dsi in range(ds_num):
                    in_factor = 2 ** dsi
                    out_factor = 2 ** (dsi + 1)
                    op.append(Conv3x3(dim * in_factor, dim * out_factor,  stride=(2, 2)))
            self.spatial_modulation.append(op)
        print("YES")
        
    def forward(self, inputs):
        out = []
        for i, feature in enumerate(inputs):
            if isinstance(self.spatial_modulation[i], nn.ModuleList):
                out_ = inputs[i]
                for III, op in enumerate(self.spatial_modulation[i]):
                    out_ = op(out_)
                out.append(out_)
            else:
                out.append(self.spatial_modulation[i](inputs[i]))
        return out
        
class TemporalModulation(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 downsample_scale=8,
                 ):
        super(TemporalModulation, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False, groups=32) 
        self.pool = nn.MaxPool3d((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

