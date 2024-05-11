from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from skimage import io
from skimage.transform import resize
import torchvision.transforms as transforms
import data.util as Util
import cv2
import torch

resize_transform = transforms.Resize((512, 512))

def textline_ext(img):
    img1 = cv2.medianBlur(img,5)
    img1 = cv2.adaptiveThreshold(img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7, 2)
    img2 = 255-img1
    kernel = np.ones((7, 7), np.uint8)
    img3 = cv2.dilate(img2,kernel,20)
    # img3 = img3 / 255.0
    return img3

class DocData2(Dataset):
    def __init__(self,dataroot,split = 'test'):
        self.split = split
        # self.imageNum = []

        self.dataPath = os.path.join(dataroot,self.split)
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
        
        data = cv2.imread(dataXPath)
        label = cv2.imread(dataYPath)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
    
        
        
        textline_dis = textline_ext(data)[:, :, np.newaxis]
        textline_gt = textline_ext(label)[:, :, np.newaxis]
      
        
        data =resize(data,(512,512))
        label = resize(label,(512,512))
        textline_dis = resize(textline_dis,(512,512))
        textline_gt = resize(textline_gt,(512,512))
        
        
        [data,label,textline_gt,textline_dis] = Util.transform_augment([data,label,textline_gt,textline_dis],split=self.split,min_max=(-1,1))
        
        
        

        return {'DIS':data,'RECT':label,'TEXT_GT':textline_gt, 'TEXT_DIS':textline_dis,'Index':index}
        
        
