# -*- coding = utf-8 -*-
# @Time : 2023/3/27 14:33
# @Author : 熊文菲
# @File : SEGDataset.py
# @Software : PyCharm
# -*- coding = utf-8 -*-
# @Time : 2023/2/20 14:07
# @Author : 熊文菲
# @File : DOCDataset.py
# @Software : PyCharm
from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io
from skimage.transform import resize
import data.util as Util
import cv2
import torch


class SEGData(Dataset):
    def __init__(self, dataroot):
        self.dataPath = os.path.join(dataroot, 'img')
        self.maskPath = os.path.join(dataroot, 'seg')
        self.dataFiles = sorted(os.listdir(self.dataPath))
        self.segFiles = sorted(os.listdir(self.maskPath))

        self.datalen = len(self.dataFiles)

    def __len__(self):
        return self.datalen

    def __getitem__(self, index):
        imagefile = self.dataFiles[index]
        segfile = self.segFiles[index]

        data = os.path.join(self.dataPath, imagefile)
        seg = os.path.join(self.maskPath, segfile)
        data = cv2.imread(data)
        label = cv2.imread(seg)
        
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]


        # 20230315改
        assert label.all() >= 0, "pixel value exists negative"
        label[label == 0] = 0.0
        label[label > 0] = 1.0

        data = cv2.resize(data, (512, 512))[:, :, np.newaxis]
        label = cv2.resize(label, (512, 512))[:, :, np.newaxis]
        # data = resize(data,(256,256),mode='constant')
        # label = resize(label,(256,256),mode='constant', order=0, preserve_range=True)

        data = data.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        data = torch.from_numpy(data).float()
        label = torch.from_numpy(label).float()

        # [data,label] = Util.transform_augment([data,label],split=self.split,min_max=(-1,1))

        return {'data': data, 'label': label}


