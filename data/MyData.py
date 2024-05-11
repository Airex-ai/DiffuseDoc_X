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
import cv2
import hdf5storage as h5
import random
from PIL import Image

from tqdm import tqdm
from torch.utils import data

class MyData(data.Dataset):
    def __init__(self,dataroot,split = 'test', img_size=288):
        self.split = split
        self.img_size = img_size
        self.dataPath = os.path.join(dataroot,self.split)
        self.disPath = os.path.join(self.dataPath, 'dis')
        self.gtPath = os.path.join(self.dataPath, 'gt')
        
        self.disFiles = sorted(os.listdir(self.disPath))
        self.gtFiles = sorted(os.listdir(self.gtPath))
        
        
        self.datalen = len(self.disFiles)

    def __len__(self):
        return self.datalen

    def __getitem__(self, index):
        # index = index % 10
        imgfile = self.disFiles[index]
        gtfile = self.gtFiles[index]
        imp = os.path.join(self.disPath, imgfile)
        gtp = os.path.join(self.gtPath, gtfile)
        
        im = np.array(Image.open(imp).convert('RGB'))[:, :, :3] / 255.
        im = cv2.resize(im, (self.img_size, self.img_size))
        im = im.transpose(2, 0, 1) # NHWC -> NCHW
        im = torch.from_numpy(im).float()
        
        gt = np.array(Image.open(gtp).convert('RGB'))[:, :, :3] / 255.
        gt = cv2.resize(gt, (self.img_size, self.img_size))
        gt = gt.transpose(2, 0, 1) # NHWC -> NCHW
        gt = torch.from_numpy(gt).float()

        return {'M':im,'F':gt,'Index':index}
        
