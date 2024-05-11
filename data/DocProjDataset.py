from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io
from skimage.transform import resize
import data.util as Util
import cv2
import torch
class DocProjDataset(Dataset):
    def __init__(self, dataroot, split):
        self.split = split
        self.dataPath = os.path.join(dataroot,self.split)
        self.disPath = os.path.join(self.dataPath, 'dis')
        self.gtPath = os.path.join(self.dataPath, 'gt')
        self.floPath = os.path.join(self.dataPath, 'flow')
        self.disFile = os.listdir(self.disPath)
        self.gtFile = os.listdir(self.gtPath)
        self.floFile = os.listdir(self.floPath)

        self.datalen = len(self.disFile)

    def __len__(self):
        return self.datalen

    def __getitem__(self, index):
        disfile_ = self.disFile[index]
        gtfile_ = self.gtFile[index]
        flofile_ = self.floFile[index]

        dis = os.path.join(self.disPath, disfile_)
        gt = os.path.join(self.gtPath, gtfile_)
        flo = os.path.join(self.floPath, flofile_)
        
        dis_ = io.imread(dis, as_gray=False).astype(np.float32)
        gt_ = io.imread(gt, as_gray=False).astype(np.float32)
        flo_ = np.load(flo)
        flo_ = flo_.astype(np.float32)
        
        dis_ = resize(dis_, (256, 256))
        gt_ = resize(gt_, (256, 256))
        
        flo_ = flo_.transpose(1, 2, 0)
        flo_ = resize(flo_, (256, 256))
        # flo_ = np.resize(flo_, (256, 256))
        
        dataXRGB = dis_
        dataYRGB = gt_

        dis_ = dis_.transpose(2, 0, 1)
        gt_ = gt_.transpose(2, 0, 1)
        flo_ = flo_.transpose(2, 0, 1)
      
        dis_ = torch.from_numpy(dis_).float()
        gt_ = torch.from_numpy(gt_).float()
        flo_ = torch.from_numpy(flo_).float()
        
        #数据增强
        # [dis_, gt_, flo_] = Util.transform_augment([dis_,gt_, flo_],split=self.split,min_max=(-1,1))

        return {'M': dis_, 'F': gt_, 'flow':flo_, 'MC':dataXRGB,'FC':dataYRGB,'nS':1,'Index':index}