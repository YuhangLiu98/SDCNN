if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import torch
import torch.nn as nn
from base import BaseModel
import torch.nn.functional as F
from torch.nn.functional import upsample, normalize
from model.Genclean import Genclean

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class SGA(nn.Module):
    def __init__(self):
        super(SGA, self).__init__()
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

    def forward(self, noise, clean):

        ca_x = self.ca(clean) * noise
        sa_x = self.sa(ca_x) * noise

        return sa_x


class GenNoise(nn.Module):
    def __init__(self, NLayer=10, FSize=64):
        super(GenNoise, self).__init__()
        kernel_size = 3
        padding = 1
        m = [nn.Conv2d(1, FSize, kernel_size=kernel_size, padding=padding),
             nn.ReLU(inplace=True)]
        for i in range(NLayer - 1):
            m.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding))
            m.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*m)

        gen_noise_w = []
        for i in range(4):
            gen_noise_w.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding))
            gen_noise_w.append(nn.ReLU(inplace=True))
        gen_noise_w.append(nn.Conv2d(FSize, 1, kernel_size=1, padding=0))
        self.gen_noise_w = nn.Sequential(*gen_noise_w)

        gen_noise_b = []
        for i in range(4):
            gen_noise_b.append(nn.Conv2d(FSize, FSize, kernel_size=kernel_size, padding=padding))
            gen_noise_b.append(nn.ReLU(inplace=True))
        gen_noise_b.append(nn.Conv2d(FSize, 1, kernel_size=1, padding=0))
        self.gen_noise_b = nn.Sequential(*gen_noise_b)
        self.SGA = SGA()
        for m in self.body:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
        for m in self.gen_noise_w:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)
        for m in self.gen_noise_b:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                nn.init.constant(m.bias, 0)

    def forward(self, x, clean, weights=None, test=False):
        noise = self.body(x)
        clean = self.body(clean)
        dep_noise = self.SGA(noise, clean)

        noise_w = self.gen_noise_w(dep_noise)
        noise_b = self.gen_noise_b(noise)

        m_w = torch.mean(torch.mean(noise_w, -1), -1).unsqueeze(-1).unsqueeze(-1)
        noise_w = noise_w - m_w
        m_b = torch.mean(torch.mean(noise_b, -1), -1).unsqueeze(-1).unsqueeze(-1)
        noise_b = noise_b - m_b

        return noise_w, noise_b


class SDCNN(BaseModel):
    def __init__(self):
        super().__init__()
        self.n_colors = 3
        FSize = 64
        self.gen_noise = GenNoise(FSize=FSize)
        self.genclean = Genclean()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        clean = self.genclean(x)
        noise_w, noise_b = self.gen_noise(x - clean, clean)
        return noise_w, noise_b, clean