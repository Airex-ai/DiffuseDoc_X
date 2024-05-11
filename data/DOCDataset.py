from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io
from skimage.transform import resize
import data.util as Util
import cv2
import torch

class DocData(Dataset):
    def __init__(self,dataroot,split = 'test'):
        self.split = split
        # self.imageNum = []

        self.dataPath = os.path.join(dataroot,split)
        self.disPath = os.path.join(self.dataPath, 'dis')
        self.gtPath = os.path.join(self.dataPath, 'gt')
        self.disFile = os.listdir(self.disPath)
        self.gtFile = os.listdir(self.gtPath)
        
        self.datalen = len(self.disFile)

    def __len__(self):
        return self.datalen

    def __getitem__(self, index):
        dataX = self.disFile[index]
        dataY = self.gtFile[index]
        
        dataXPath = os.path.join(self.disPath,dataX)
        dataYPath = os.path.join(self.gtPath,dataY)
        
        data = io.imread(dataXPath,as_gray=True).astype(float)[:,:,np.newaxis]
        label = io.imread(dataYPath,as_gray=True).astype(float)[:,:,np.newaxis]
        
        
        data = resize(data,(512,512))
        label = resize(label,(512,512))
        
        dataXRGB = io.imread(dataXPath).astype(float)
        dataYRGB = io.imread(dataYPath).astype(float)
        
        dataXRGB = resize(dataXRGB,(512,512))
        dataYRGB = resize(dataYRGB,(512,512))
        
        
        
        
        [data,label] = Util.transform_augment([data,label],split=self.split,min_max=(-1,1))
        
        

        return {'M':data,'F':label,'MC':dataXRGB,'FC':dataYRGB,'nS':1,'Index':index}
        
        


