import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb
from collections import OrderedDict
from einops import rearrange

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class CMM(nn.Module):
    def __init__(self, inp, T=7,num_heads=16):
        super(CMM, self).__init__()
        self.T = T
        self.inp = inp
        self.reduce_time = num_heads

        
        self.qkv = nn.Linear(inp, inp * 3, bias=False)
        self.qconv = nn.Conv2d(inp//self.reduce_time, inp//self.reduce_time, 3, 1, 1, groups=inp//self.reduce_time, bias=False)
        self.kconv = nn.Conv2d(inp//self.reduce_time, inp//self.reduce_time, 3, 1, 1, groups=inp//self.reduce_time, bias=False)

        self.mlp=nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(inp,inp,bias=False)),
            ('act', nn.GELU()),
            ('fc2', nn.Linear(inp,inp,bias=False)),
            ('act2', nn.Sigmoid())
        ]))

    def forward(self, input):
        # build shift conv layer
        input = input.permute(0,1,3,4,2).contiguous()
        B, T, H, W, C0 = input.shape
        qkv = self.qkv(input).view(B, T, H, W, 3, self.reduce_time, C0 // self.reduce_time)
        q, k, v = qkv[:,:,:,:,0], qkv[:,:,:,:,1], qkv[:,:,:,:,2]  # B, T, H, W, N, C
        # print('q:{}'.format(q.shape))
        q = q.permute(0,4,1,5,2,3).contiguous().view(-1, C0 // self.reduce_time,H, W) # B, N, T, C, H, W -> BNT,C, H, W
        k = k.permute(0,4,1,5,2,3).contiguous().view(-1,C0 // self.reduce_time,H, W)   # B, N, T, C, H, W-> BNT,C, H, W
        # print('q1:{}'.format(q.shape))
        q = self.qconv(q)
        # print('q2:{}'.format(q.shape))
        C = C0 // self.reduce_time
        q = q.view(-1, T, C, H, W)
        # print('q3:{}'.format(q.shape))
        k = self.kconv(k)
        k = k.view(-1, T, C, H, W)
        dshift1 = k[:, 1:, :, :, :]
        R = dshift1 - q[:,:-1]
        R = F.pad(R, (0,0,0,0,0,0,0,1,0,0))
        R = R.view( B, -1, T, C, H, W)
        # print('x3:{}'.format(R.shape))
        R = R.permute(0,2,4,5,1,3).contiguous().view(B,T,H,W,-1)
        # R = rearrange(R, 'b n t c h w -> b t h w n c').contiguous().view(B,T,H,W,-1)
        R = self.mlp(R)
        v = R * v.view(B,T,H,W,-1) #(5,8,7,7,2048)
        # print('x4:{}'.format(R.shape))
        v = v.permute(0,1,4,2,3).contiguous()
        return -v

