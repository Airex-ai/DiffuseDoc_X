# -*- coding = utf-8 -*-
# @Time : 2023/3/27 14:40
# @Author : 熊文菲
# @File : loss.py
# @Software : PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
import cv2
import matplotlib.pyplot as plt
import pytorch_ssim

class gradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dH) + torch.mean(dW)) / 2.0
        return loss


class crossCorrelation2D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9), voxel_weights=None):
        super(crossCorrelation2D, self).__init__()
        self.kernel = kernel
        self.voxel_weight = voxel_weights
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1]])).cuda()

    def forward(self, input, target):

        min_max = (-1, 1)
        target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        II = input * input
        TT = target * target
        IT = input * target

        pad = (int((self.kernel[0] - 1) / 2), int((self.kernel[1] - 1) / 2))
        T_sum = F.conv2d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv2d(input, self.filt, stride=1, padding=pad)
        TT_sum = F.conv2d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv2d(II, self.filt, stride=1, padding=pad)
        IT_sum = F.conv2d(IT, self.filt, stride=1, padding=pad)
        kernelSize = self.kernel[0] * self.kernel[1]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        # cross = (I-Ihat)(J-Jhat)
        cross = IT_sum - Ihat * T_sum - That * I_sum + That * Ihat * kernelSize
        T_var = TT_sum - 2 * That * T_sum + That * That * kernelSize
        I_var = II_sum - 2 * Ihat * I_sum + Ihat * Ihat * kernelSize

        cc = cross * cross / (T_var * I_var + 1e-5)
        loss = -1.0 * torch.mean(cc)
        return loss
    
def unwarp(img, bm):
    
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


class Unwarploss(torch.nn.Module):
    def __init__(self):
        super(Unwarploss, self).__init__()
        # self.xmx, self.xmn, self.ymx, self.ymn = 166.28639310649825, -3.792634897181367, 189.04606710275974, -18.982843029373125
        # self.xmx, self.xmn, self.ymx, self.ymn = 434.8578833991327, 14.898654260467202, 435.0363953546216, 14.515746051497239
        # self.xmx, self.xmn, self.ymx, self.ymn = 434.9877152088082, 14.546402972133514, 435.0591952709043, 14.489902537540008
        # self.xmx, self.xmn, self.ymx, self.ymn = 435.14545757153445, 13.410177297916455, 435.3297804574046, 14.194541402379988
        self.xmx, self.xmn, self.ymx, self.ymn = 0.0,0.0,0.0,0.0

        

    def forward(self,imgout,imggt):
        #image [n,c,h,w], target_nhwc [n,h,w,c], labels [n,h,w,c]
        # n,c,h,w=inp.shape           #this has 6 channels if image is passed
        # print (h,w)
        # inp=inp.detach().cpu().numpy()
        # inp_img=inp[:,:3,:,:] #img in bgr 

        # denormalize pred
        # pred=(pred/2.0)+0.5
        # pred[:,:,:,0]=(pred[:,:,:,0]*(self.xmx-self.xmn)) +self.xmn
        # pred[:,:,:,1]=(pred[:,:,:,1]*(self.ymx-self.ymn)) +self.ymn
        # pred[:,:,:,0]=pred[:,:,:,0]/float(448.0)
        # pred[:,:,:,1]=pred[:,:,:,1]/float(448.0)
        # pred=(pred-0.5)*2
        # pred=pred.double()

        # denormalize label
        # label=(label/2.0)+0.5
        # label[:,:,:,0]=(label[:,:,:,0]*(self.xmx-self.xmn)) +self.xmn
        # label[:,:,:,1]=(label[:,:,:,1]*(self.ymx-self.ymn)) +self.ymn
        # label[:,:,:,0]=label[:,:,:,0]/float(448.0)
        # label[:,:,:,1]=label[:,:,:,1]/float(448.0)
        # label=(label-0.5)*2
        # label=label.double()

        # imgout=unwarp(inp_img,pred)
        # imggt=unwarp(inp_img,label)
        loss_fn = nn.MSELoss(reduction='mean')
        ssim_loss = pytorch_ssim.SSIM()
        uloss=loss_fn(imgout,imggt)
        ssim = 1-ssim_loss(imgout,imggt)

        # print(uloss)
        # del pred
        # del label
        # del inp

        return uloss.float(),ssim.float()