import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import hdf5storage as h5
import random

from tqdm import tqdm
from torch.utils import data

class Doc3D(data.Dataset):
    def __init__(self,dataroot,split = 'test',is_transform = False, img_size=288):
        self.split = split
        self.img_size = img_size
        self.is_transform = is_transform
        self.dataPath = os.path.join(dataroot,self.split)
        self.disPath = os.path.join(self.dataPath, 'img')
        self.gtPath = os.path.join(self.dataPath, 'gt')
        self.bmPath = os.path.join(self.dataPath, 'bm')
        self.wcPath = os.path.join(self.dataPath, 'wc')
        self.disFiles = []
        self.gtFiles = []
        self.bmFiles = []
        self.wcFiles = []
        for dis, gt , bm, wc in zip (sorted(os.listdir(self.disPath)),sorted(os.listdir(self.gtPath)),sorted(os.listdir(self.bmPath)),sorted(os.listdir(self.wcPath))): # 1,2,3,4...
            for dis_, gt_, bm_, wc_ in zip(sorted(os.listdir(os.path.join(self.disPath, dis))),sorted(os.listdir(os.path.join(self.gtPath, gt))),sorted(os.listdir(os.path.join( self.bmPath, bm))),sorted(os.listdir(os.path.join(self.wcPath, wc)))):#*_***.png
                self.disFiles.append(os.path.join(self.disPath, dis, dis_))
                self.gtFiles.append(os.path.join(self.gtPath, gt, gt_))
                self.bmFiles.append(os.path.join(self.bmPath, bm, bm_))
                self.wcFiles.append(os.path.join(self.wcPath, wc, wc_))
                
        self.disFiles = sorted(self.disFiles)
        self.gtFiles = sorted(self.gtFiles)
        self.bmFiles = sorted(self.bmFiles)
        self.wcFiles = sorted(self.wcFiles)
        # self.disFiles = sorted(os.listdir(self.disPath))
        # self.gtFiles = sorted(os.listdir(self.gtPath))
        # self.bmFiles = sorted(os.listdir(self.bmPath))
        # self.wcFiles = sorted(os.listdir(self.wcPath))
        self.datalen = len(self.disFiles)

    def __len__(self):
        return self.datalen

    def __getitem__(self, index):
        # index = index % 10
        imgfile = self.disFiles[index]
        gtfile = self.gtFiles[index]
        bmfile = self.bmFiles[index]
        wcfile = self.wcFiles[index]
        # print(t)
        # imgPath = os.path.join(self.disPath,imgfile)
        # gtPath = os.path.join(self.gtPath,gtfile)
        # bmPath = os.path.join(self.bmPath,bmfile)
        # wcPath = os.path.join(self.wcPath,wcfile)
        
        
        im = cv2.imread(imgfile).astype(np.float32)
        im = im[..., ::-1]
        
        gt = cv2.imread(gtfile).astype(np.float32)
        gt = gt[..., ::-1]

        
        wc = cv2.imread(wcfile, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED).astype(np.float32)

        bm = h5.loadmat(bmfile)['bm']

        if self.is_transform:
            im, gt, lbl = self.transform(wc,bm,im,gt)
        return {'M':im,'F':gt, 'BM':lbl,'Index':index}


    def tight_crop(self, wc, im):
        msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8)
        size=msk.shape
        [y, x] = (msk).nonzero()
        minx = min(x)
        maxx = max(x)
        miny = min(y)
        maxy = max(y)
        wc = wc[miny : maxy + 1, minx : maxx + 1, :]
        im = im[miny : maxy + 1, minx : maxx + 1, :]
        
        s = 20
        wc = np.pad(wc, ((s, s), (s, s), (0, 0)), 'constant')
        im = np.pad(im, ((s, s), (s, s), (0, 0)), 'constant')
        cx1 = random.randint(0, s - 5)
        cx2 = random.randint(0, s - 5) + 1
        cy1 = random.randint(0, s - 5)
        cy2 = random.randint(0, s - 5) + 1

        wc = wc[cy1 : -cy2, cx1 : -cx2, :]
        im = im[cy1 : -cy2, cx1 : -cx2, :]
        t=miny-s+cy1
        b=size[0]-maxy-s+cy2
        l=minx-s+cx1
        r=size[1]-maxx-s+cx2

        return wc,im,t,b,l,r


    def transform(self, wc, bm, im, gt):
        wc,im,t,b,l,r=self.tight_crop(wc,im)               #t,b,l,r = is pixels cropped on top, bottom, left, right
        im = cv2.resize(im, (self.img_size, self.img_size))
        # im = im[:, :, ::-1] # RGB -> BGR
        im = im.astype(np.float64)
        if im.shape[2] == 4:
            im=im[:,:,:3]
        im = im.astype(float) / 255.0
        im = im.transpose(2, 0, 1) # NHWC -> NCHW
       
        # msk=((wc[:,:,0]!=0)&(wc[:,:,1]!=0)&(wc[:,:,2]!=0)).astype(np.uint8) * 255
        # #normalize label
        # xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497
        # wc[:,:,0]= (wc[:,:,0]-zmn)/(zmx-zmn)
        # wc[:,:,1]= (wc[:,:,1]-ymn)/(ymx-ymn)
        # wc[:,:,2]= (wc[:,:,2]-xmn)/(xmx-xmn)
        # wc=cv2.bitwise_and(wc,wc,mask=msk)
        
        # wc = m.imresize(wc, self.img_size) 
        # wc = wc.astype(float) / 255.0
        # wc = wc.transpose(2, 0, 1) # NHWC -> NCHW

        bm = bm.astype(float)
        #normalize label [-1,1]
        bm[:,:,1]=bm[:,:,1]-t
        bm[:,:,0]=bm[:,:,0]-l
        bm=bm/np.array([448.0-l-r, 448.0-t-b])
        bm=(bm-0.5)*2

        bm0=cv2.resize(bm[:,:,0],(self.img_size,self.img_size))
        bm1=cv2.resize(bm[:,:,1],(self.img_size,self.img_size))
        
        gt = cv2.resize(gt, (self.img_size, self.img_size))
        # gt = gt[:, :, ::-1] # RGB -> BGR
        gt = gt.astype(float) / 255.0
        gt = gt.transpose(2, 0, 1) # NHWC -> NCHW
        
        # img=np.concatenate([alb,wc],axis=0)
        lbl=np.stack([bm0,bm1],axis=-1)
        lbl = lbl.transpose(2,0,1)

        img = torch.from_numpy(im).float()
        gt = torch.from_numpy(gt).float()
        lbl = torch.from_numpy(lbl).float()
        return img, gt, lbl
