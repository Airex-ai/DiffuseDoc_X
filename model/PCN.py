import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.spectral_norm import spectral_norm

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type, use_sn=use_sn)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type, use_sn=use_sn)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class DilationBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(DilationBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 2, norm=norm, activation=activation, pad_type=pad_type, dilation=2)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 4, norm=norm, activation=activation, pad_type=pad_type, dilation=4)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 8, norm=norm, activation=activation, pad_type=pad_type, dilation=8)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', dilation=1,
                 use_bias=True, use_sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if use_sn:
            self.conv = spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias, dilation=dilation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransConv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu'):
        super(TransConv2dBlock, self).__init__()
        self.use_bias = True

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'in_affine':
            self.norm = nn.InstanceNorm2d(norm_dim, affine=True)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.transConv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, bias=self.use_bias)

    def forward(self, x):
        x = self.transConv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
          self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
          self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
          return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
          return F.layer_norm(x, normalized_shape)

class DownsampleResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='in', activation='relu', pad_type='zero', use_sn=False):
        super(DownsampleResBlock, self).__init__()
        self.conv_1 = nn.ModuleList()
        self.conv_2 = nn.ModuleList()

        self.conv_1.append(Conv2dBlock(input_dim,input_dim,3,1,1,'none',activation,pad_type,use_sn=use_sn))
        self.conv_1.append(Conv2dBlock(input_dim,output_dim,3,1,1,'none',activation,pad_type,use_sn=use_sn))
        self.conv_1.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv_1 = nn.Sequential(*self.conv_1)


        self.conv_2.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv_2.append(Conv2dBlock(input_dim,output_dim,1,1,0,'none',activation,pad_type,use_sn=use_sn))
        self.conv_2 = nn.Sequential(*self.conv_2)


    def forward(self, x):
        out = self.conv_1(x) + self.conv_2(x)
        return out


class FlowColumn(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=2, activ='lrelu',
                 norm='in', pad_type='reflect', use_sn=True):
        super(FlowColumn, self).__init__()

        self.down_flow01 = nn.Sequential(
            Conv2dBlock(input_dim, dim // 2, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim // 2, dim // 2, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        self.down_flow02 = nn.Sequential(
            Conv2dBlock(dim // 2, dim // 2, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        self.down_flow03 = nn.Sequential(
            Conv2dBlock(dim // 2, dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        self.down_flow04 = nn.Sequential(
            Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        self.down_flow05 = nn.Sequential(
            Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        self.down_flow06 = nn.Sequential(
            Conv2dBlock(4 * dim, 8 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))

        dim = 8 * dim

        self.up_flow05 = nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim, dim // 2, 6, 2, 2, norm=norm, activation=activ))

        self.up_flow04 = nn.Sequential(
            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 2, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 2, dim // 4, 6, 2, 2, norm=norm, activation=activ))

        self.up_flow03 = nn.Sequential(
            Conv2dBlock(dim // 2, dim // 4, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 4, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 4, dim // 8, 6, 2, 2, norm=norm, activation=activ))

        self.up_flow02 = nn.Sequential(
            Conv2dBlock(dim // 4, dim // 8, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 8, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 8, dim // 16, 6, 2, 2, norm=norm, activation=activ))

        self.up_flow01 = nn.Sequential(
            Conv2dBlock(dim // 8, dim // 16, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 16, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 16, dim // 16, 6, 2, 2, norm=norm, activation=activ))

        self.location = nn.Sequential(
            Conv2dBlock(dim // 8, dim // 16, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 16, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 16, dim // 16, 6, 2, 2, norm=norm, activation=activ),
            Conv2dBlock(dim // 16, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False))

        # self.to_flow05 = Conv2dBlock(dim // 2, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)
        # self.to_flow04 = Conv2dBlock(dim // 4, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)
        # self.to_flow03 = Conv2dBlock(dim // 8, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)
        # self.to_flow02 = Conv2dBlock(dim // 16, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)
        # self.to_flow01 = Conv2dBlock(dim // 16, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)

    def unwarp(self, img, bm):
    
        # print(bm.type)
        # img=torch.from_numpy(img).cuda().double()
        n,c,h,w=img.shape
        # resize bm to img size
        # print bm.shape
        bm = bm.transpose(3, 2).transpose(2, 1)
        bm = F.upsample(bm, size=(h, w), mode='bilinear')
        bm = bm.transpose(1, 2).transpose(2, 3)
        # print bm.shape

        img=img.double()
        # print(img.type)
        res = F.grid_sample(input=img, grid=bm)
        # print(res.shape)
        return res
    
    def forward(self, inputs):
        f_x1 = self.down_flow01(inputs)
        f_x2 = self.down_flow02(f_x1)
        f_x3 = self.down_flow03(f_x2)
        f_x4 = self.down_flow04(f_x3)
        f_x5 = self.down_flow05(f_x4)
        f_x6 = self.down_flow06(f_x5)

        f_u5 = self.up_flow05(f_x6)
        f_u4 = self.up_flow04(torch.cat((f_u5, f_x5), 1))
        f_u3 = self.up_flow03(torch.cat((f_u4, f_x4), 1))
        f_u2 = self.up_flow02(torch.cat((f_u3, f_x3), 1))
        f_u1 = self.up_flow01(torch.cat((f_u2, f_x2), 1))
        flow_map = self.location(torch.cat((f_u1, f_x1), 1))

        # flow05 = self.to_flow05(f_u5)
        # flow04 = self.to_flow04(f_u4)
        # flow03 = self.to_flow03(f_u3)
        # flow02 = self.to_flow02(f_u2)
        # flow01 = self.to_flow01(f_u1)

        # return flow_map, [flow01, flow02, flow03, flow04, flow05]
        out = self.unwarp(input,flow_map)
        return out, flow_map