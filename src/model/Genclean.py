if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)

        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out

class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):

        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out

class Genclean(nn.Module):
    def __init__(self, in_ch=1, out_ch=64, sobel_ch=64):
        super(Genclean, self).__init__()
        self.conv0 = ConvLayer(in_ch, sobel_ch, 1, 1)

        self.conv_p1 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p3 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p4 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p5 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f5 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p6 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f6 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p7 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f7 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p8 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f8 = nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1)
        # *****************************************************************************************
        self.conv_sobel = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_p1_e = nn.Conv2d(sobel_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f1_e = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p2_e = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f2_e = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p3_e = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f3_e = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p4_e = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f4_e = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p5_e = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f5_e = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p6_e = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f6_e = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p7_e = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f7_e = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_f8_e = nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out_0_log = self.conv_sobel(x)  # 网络结构

        out_1_log = self.relu(self.conv_p1_e(out_0_log))
        out_1_log = self.relu(self.conv_f1_e(out_1_log))

        out_2_log = self.relu(self.conv_p2_e(out_1_log))
        out_2_log = self.relu(self.conv_f2_e(out_2_log))

        out_3_log = self.relu(self.conv_p3_e(out_2_log))
        out_3_log = self.relu(self.conv_f3_e(out_3_log))

        out_4_log = self.relu(self.conv_p4_e(out_3_log))
        out_4_log = self.relu(self.conv_f4_e(out_4_log))

        out_5_log = self.relu(self.conv_p5_e(out_4_log))
        out_5_log = self.relu(self.conv_f5_e(out_5_log))

        out_6_log = self.relu(self.conv_p6_e(out_5_log))
        out_6_log = self.relu(self.conv_f6_e(out_6_log))

        out_7_log = self.relu(self.conv_p7_e(out_6_log))
        out_7_log = self.relu(self.conv_f7_e(out_7_log))
        
        # ******************************************
        out_0 = self.conv0(x)  # 网络结构

        out_1 = self.relu(self.conv_p1(out_0 + out_0_log))
        out_1 = self.relu(self.conv_f1(out_1))

        out_2 = self.relu(self.conv_p2(out_1 + out_1_log))
        out_2 = self.relu(self.conv_f2(out_2))

        out_3 = self.relu(self.conv_p3(out_2 + out_2_log))
        out_3 = self.relu(self.conv_f3(out_3))

        out_4 = self.relu(self.conv_p4(out_3 + out_3_log))
        out_4 = self.relu(self.conv_f4(out_4))

        out_5 = self.relu(self.conv_p5(out_4 + out_4_log))
        out_5 = self.relu(self.conv_f5(out_5))

        out_6 = self.relu(self.conv_p6(out_5 + out_5_log))
        out_6 = self.relu(self.conv_f6(out_6))

        out_7 = self.relu(self.conv_p7(out_6 + out_6_log))
        out_7 = self.relu(self.conv_f7(out_7))

        out_8 = self.relu(self.conv_p8(out_7 + out_7_log))
        out_8 = self.conv_f8(out_8)
        out = self.relu(out_8)

        return out
