import os
import cv2
import time
import torch
import skimage
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data
import torch.optim as optim
import core.logger as Logger
import torch.nn.functional as F
from skimage.transform import resize

from skimage import io
from model.seg import U2NETP
from torch.autograd import Variable
from model.metrics import tensor2im


def feed_data_to_device(x):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if isinstance(x, dict):
        for key, item in x.items():
            if item is not None and not isinstance(item, list):
                x[key] = item.to(device)
    elif isinstance(x, list):
        for item in x:
            if item is not None:
                item = item.to(device)
    else:
        x = x.to(device)
    return x


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default='./Documet/seg_val/val1/')
    parser.add_argument('--resultsPath', type=str, default='./results/')
    parser.add_argument('--modelRoot', type=str, default='./segmodel/1678883099.6107142/59seg.pth')
    args = parser.parse_args()

    model_dir = args.modelRoot

    results = os.path.join(args.resultsPath, '{}'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
    if not os.path.exists(results):
        os.makedirs(results)

    # load model
    model = U2NETP(3, 1)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_dir))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_dir, map_location='cpu'))
    model.eval()

    # load data
    datapath = args.dataPath
    for data in os.listdir(datapath):
        dataName = os.path.join(datapath, data)
        img = cv2.imread(dataName)

        img = resize(img, (256, 256))
        img1 = img.transpose(2, 0, 1)
        img1 = torch.from_numpy(img1).float().unsqueeze(0)
        # img = Variable()
        img1 = feed_data_to_device(img1)

        d0, d1, d2, d3, d4, d5, d6 = model(img1)

        d0_0 = d0[:, 0, :, :]
        d0_0 = normPRED(d0_0)

        d0_1 = d0_0.squeeze()
        d0_2 = d0_1.cpu().data.numpy()
        # d0_2[d0_2 > 0.005] = 1
        # d0_2[d0_2 <= 0.005] = 0
        mask = Image.fromarray(d0_2 * 255).convert('RGB')

        # img_x = img_x.squeeze()
        # # print(img_x.size())
        # img_x = img_x.cpu().float()
        # img_x = img_x.numpy()
        # img_x = img_x.transpose(1,2,0)
        # # print(img_x.shape)
        image = io.imread(dataName)
        imo = mask.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
        pb_np = np.array(imo)

        savename = os.path.join(results, data)

        imo.save(savename)



