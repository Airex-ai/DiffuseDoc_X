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
from skimage.transform import resize

from skimage import io
from model.rect import registUnetBlock
from torch.autograd import Variable
from model.metrics import tensor2im
from model.networks import init_weights

from model.seg import U2NETP
from model import loss
from model import gradient_loss

from tensorboardX import SummaryWriter

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)
    
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def textline_extractor(x):
        # x = tensor2im(x.detach().float().cpu(),min_max=(0,1)) #(h,w,c)(512,512,1)
        # x = x.transpose(2,0,1)[np.newaxis,:,:,:]
        batch_output = np.zeros_like(x)#b,c,h,w
        x = x.detach().cpu().numpy()
        x = x.transpose(0, 2, 3, 1)  # 将 channel 维度放到最后，以符合 cv2 的格式要求 b,h,w,c
        x = x.astype(np.uint8)
        # b,c,h,w = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        # batch_output = np.zeros((b,1,h,w))
        for i, img in enumerate(x):
            # print(img.shape)
            
            # img = np.transpose(img,(1,2,0))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# h,w,c
            # print(img.shape)
            img = cv2.medianBlur(img, 3)
            # print(img.shape)
            img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,3, 2)
            img = 255-img
            kernel = np.ones((3, 3), np.uint8)
            img = cv2.dilate(img,kernel,20)
            img = img[:, :, np.newaxis]
            img = np.transpose(img,(2,0,1))
            # img = img / 255.0
            batch_output[i] = img
        # x = cv2.medianBlur(x,5)
        # x = cv2.adaptiveThreshold(x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7, 2)
        # x = 255-x
        # kernel = np.ones((7, 7), np.uint8)
        # x = cv2.dilate(x,kernel,20)
        batch_output=torch.from_numpy(batch_output).float().cuda()
        return batch_output

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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/media/wit/HDD_0/xwf/data/')
    parser.add_argument('--phase',type=str,default='train')
    # parser.add_argument('--datapathTest', type=str, default='/media/wit/HDD_0/xwf/DiffuseDoc/Documet/seg/test/')
    parser.add_argument('--epochs', type=int, default=40, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LN')
    parser.add_argument('--batchsize', type=int, default=1, metavar='N')
    parser.add_argument('--savemodelPath', type=str, default='./models/rect/')
    parser.add_argument('--use_shuffle', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8, metavar='N')
    parser.add_argument('--parallel', default=False, type=str2bool, help='Set to True to train on parallel GPUs')
    parser.add_argument('--beta1', default=0.9, help='Beta Values for Adam Optimizer')
    parser.add_argument('--beta2', default=0.999, help='Beta Values for Adam Optimizer')
    parser.add_argument('--logroot', type=str, default='./logs/rectlog/')
    parser.add_argument('--imgsave', type=str, default='./rectimages/')
    parser.add_argument('-gpu','--gpu_ids',type=str,default=None)
    parser.add_argument('--rectpath',type=str,default=None)
    parser.add_argument('--imagepath',type=str,default='')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    
    from data.DOCData2 import DocData2 as D
    if args.phase == 'train':
        train_dataset = D(dataroot=args.datapath, split=args.phase)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize,
                                               shuffle=args.use_shuffle, num_workers=args.num_workers)
    else:    
        test_dataset = D(dataroot=args.datapath, split=args.phase)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize,
                                              shuffle=args.use_shuffle, num_workers=args.num_workers)
        
    
    seg_model = U2NETP(1,1)
    segdir = '/media/wit/HDD_0/xwf/DiffuseDocV2/models/segmodel/2023-04-20 13:22:25/39seg.pth'
    seg_model.load_state_dict(torch.load(segdir))
    seg_model.cuda()
    seg_model.eval()
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    timenow = '{}'.format(time.strftime("%Y-%m-%d %H:%M:%S"))
    
        
    if args.phase == 'train': 
        
        tensorboader_dir = os.path.join(args.logroot, 'tensorboder/{}'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
        if not os.path.exists(tensorboader_dir):
            os.makedirs(tensorboader_dir)
        
        results = SummaryWriter(tensorboader_dir)
        
        # model = registUnetBlock(3,[32,32,64,128,256,512,1024],[1024,512,256,128,64,32,32,2])
        model = registUnetBlock(3,[64,128,256,512,1024],[1024,512,256,128,64,2])
        init_weights(model, init_type='normal')

        if args.parallel:
            model = torch.nn.DataParallel(model)
        else:
            model = model.cuda()
            
        optimizer = optim.Adam(model.parameters(),
                           lr=float(args.lr),
                           betas=(float(args.beta1), float(args.beta2)))
        
        Loss1 = nn.MSELoss(reduction='mean').to(device)
        # Loss2 = nn.BCELoss(size_average=True).to(device)
        # Loss1 = nn.BCELoss(size_average=True).to(device)
        Loss2 = loss.crossCorrelation2D(1, kernel=(9, 9)).to(device)
        # Loss2 = nn.L1Loss(reduction='mean').to(device)
        Loss3 = loss.gradientLoss("l2").to(device)
        # Loss3 = gradient_loss.GradientPriorLoss().to(device)

        modeldir = args.savemodelPath + '{}/'.format(timenow)
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
         
        for epoch in range(args.epochs):
            if isinstance(model, torch.nn.Module):
                model.train()
            for step, train_data in enumerate(tqdm(train_loader)):
                train_data = feed_data_to_device(train_data)
                iters = epoch * len(train_loader) + step
                
                # x_d = train_data['DIS']
                # x_line = train_data['TEXT']
                x_d = train_data['DIS'] * 255
                mask_reg, _, _, _, _, _, _, =seg_model(x_d)
                # textline = textline_extractor(train_data['DIS'])
                # mask_reg = (mask_reg > 0.5).float()
                rect,flow,textline_rect,mask_rect = model(torch.cat([train_data['DIS'],train_data['TEXT_DIS'], mask_reg], dim=1))
                
                # l_flow = Loss2(flow)
                
                mask = torch.ones_like(mask_rect, requires_grad=False).cuda()
                l_pix = Loss2(rect, train_data['RECT'])
                
                l_mask = Loss1(mask_rect, mask) 
                
                l_textline = Loss2(textline_rect, train_data['TEXT_GT']) 
                # l_textline = Loss3(rect, train_data['RECT']) * 1e-4
                
                l_flow = Loss3(flow)
                
                # l_tot = l_mask + l_textline + l_pix 
                l_tot = l_pix + l_mask + l_textline + l_flow
                
                optimizer.zero_grad()
                l_tot.requires_grad_(True)
                l_tot.backward()
                optimizer.step()
                
                results.add_scalar('loss/l_pix', float(l_pix.data), global_step=iters)
                results.add_scalar('loss/l_mask', float(l_mask.data), global_step=iters)
                results.add_scalar('loss/l_textline', float(l_textline.data), global_step=iters)
                results.add_scalar('loss/l_flow', float(l_flow.data), global_step=iters)
                results.add_scalar('loss/l_tot', float(l_tot.data), global_step=iters)
                
                print("epoch: [{}/{}][{}/{}] l_pix:{} l_mask:{} l_textline:{} l_flow:{} l_tot:{} \n".format(epoch+1, args.epochs, iters, len(train_loader)* args.epochs, l_pix.data, l_mask.data, l_textline.data, l_flow.data, l_tot.data))
                # print("epoch: [{}/{}][{}/{}]\t l_mask: {}  l_textline: {}  l_tot: {}  ".format(epoch+1, args.epochs, iters, len(train_loader) * args.epochs, l_mask.data, l_textline.data, l_tot.data))
                if(iters % (3770 * 5 )== 0):
                    rect_img = tensor2im(rect.detach().float().cpu(),min_max=(0,1))
                    flow_img = tensor2im(flow.detach().float().cpu(),min_max=(0,1))
                    textline_img = tensor2im(textline_rect.detach().float().cpu(),min_max=(0,1))
                    textline_gt = tensor2im(train_data['TEXT_GT'].detach().float().cpu(),min_max=(-1,1))
                    dis_input = tensor2im(train_data['DIS'].detach().float().cpu(),min_max=(-1,1))
                    gt = tensor2im(train_data['RECT'].detach().float().cpu(),min_max=(-1,1))
                    # mask_rect = tensor2im(mask_rect.detach().float().cpu(),min_max=(0,1))
                    
                    
                    imgpath = os.path.join(args.imgsave,'{}'.format(timenow))
                    if not os.path.exists(imgpath):
                        os.makedirs(imgpath)
                    savePath1 = os.path.join(imgpath, '{}_grey_rect.png'.format(iters))
                    save_image(rect_img,savePath1)
                    
                    savePath2 = os.path.join(imgpath, '{}_flow_rect.png'.format(iters))
                    save_image(flow_img,savePath2)
                    
                    savePath3 = os.path.join(imgpath, '{}_textline_rect.png'.format(iters))
                    save_image(textline_img,savePath3)
                    
                    savePath4 = os.path.join(args.imgsave,'{}'.format(timenow), '{}_textline_gt.png'.format(iters))
                    save_image(textline_gt,savePath4) 
                    
                    savePath5 = os.path.join(args.imgsave,'{}'.format(timenow), '{}_dis_input.png'.format(iters))
                    save_image(dis_input,savePath5) 
                    
                    savePath6 = os.path.join(args.imgsave,'{}'.format(timenow), '{}_gt.png'.format(iters))
                    save_image(gt,savePath6) 
                    
                    savePath7 = os.path.join(args.imgsave,'{}'.format(timenow), '{}_mask.png'.format(iters))
                    # save_image(mask_rect,savePath7)
                    d0_0 = mask_rect[:, 0, :, :]
                    d0_0 = normPRED(d0_0)
                    d0_1 = d0_0.squeeze()
                    d0_2 = d0_1.cpu().data.numpy()
                    d0_3 = Image.fromarray(d0_2 * 255).convert('RGB')
                    d0_3.save(savePath7)
                    
            if(epoch % 5 == 0):        
                modelpath = modeldir + '{}_rect.pth'.format(epoch)
                print('saving model......')
                torch.save(model.state_dict(), modelpath)
                print('done..............')
                
        modelpath = modeldir + 'rect.pth'
        print('saving model......')
        torch.save(model.state_dict(), modelpath)
        print('all done..............')
        
    # else:
    #     model = registUnetBlock(3,[32,32,64,128,256,512,1024],[1024,512,256,128,64,32,32,2])
    #     modeldir = args.rectpath
    #     assert os.path.exists(modeldir)
        
    #     if torch.cuda.is_available():
    #         model.load_state_dict(torch.load(modeldir))
    #         model.cuda()
    #     else:
    #         model.load_state_dict(torch.load(modeldir, map_location='cpu'))
        
    #     model = model.eval()
        
    #     assert os.path.exists(args.imagepath)
    #     for data in os.listdir(args.imagepath):
    #         dataName = os.path.join(args.imagepath, data)
    #         img = cv2.imread(dataName)
    #         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
            
    #         img1 = resize(gray_img, (512,512))
    #         img1 = img1.transpose(2, 0, 1)
    #         img1 = torch.from_numpy(img1).float().unsqueeze(0)
    #         img1 = feed_data_to_device(img1)
            
    #         mask_reg, _, _, _, _, _, _, =seg_model(img1)
    #         # mask_reg = (mask_reg > 0.5).float()
    #         rect,flow,textline,mask_rect = model(torch.cat([img1, mask_reg], dim=1))
            
    #         rect_img = tensor2im(rect.detach().float().cpu(),min_max=(0,1))
    #         flow_img = tensor2im(flow.detach().float().cpu(),min_max=(0,1))
    #         textline_img = tensor2im(textline.detach().float().cpu(),min_max=(0,1))
            
    #         d0_0 = mask_rect[:, 0, :, :]
    #         d0_0 = normPRED(d0_0)
    #         d0_1 = d0_0.squeeze()
    #         d0_2 = d0_1.cpu().data.numpy()
    #         mask = Image.fromarray(d0_2 * 255).convert('RGB')
    #         # d0_0[d0_0 > 0.5] = 1
    #         # d0_0[d0_0 <= 0.5] = 0
    #         # mask = d0_0.unsqueeze(0)
            
    #         # mask_rect_img = tensor2im(mask.detach().float().cpu(),min_max=(0,1))
            
            
    #         savePath1 = os.path.join(args.imgsave,'{}'.format(timenow), 'grey_rect_{}'.format(data))
    #         save_image(rect_img,savePath1)
            
    #         savePath2 = os.path.join(args.imgsave,'{}'.format(timenow), 'flow_rect_{}'.format(data))
    #         save_image(flow_img,savePath2)
            
    #         savePath3 = os.path.join(args.imgsave,'{}'.format(timenow), 'textline_rect_{}'.format(data))
    #         save_image(textline_img,savePath3)
            
    #         savePath4 = os.path.join(args.imgsave,'{}'.format(timenow), 'mask_rect_{}'.format(data))
    #         mask.save(savePath4)
            
    #         #保存彩色图像
    #         from model.rect import Dense2DSpatialTransformer
    #         import torch.nn.functional as F
    #         stn = Dense2DSpatialTransformer()
    #         h = img.shape[0]
    #         w = img.shape[1]
    #         img = resize(img,(512,512))
    #         img = img.transpose(2,0,1)
    #         img = torch.from_numpy(img).float().unsqueeze(0)
    #         img.to(flow.device)
    #         out1 = stn(img[:, 0:1],flow)
    #         out2 = stn(img[:, 1:2],flow)
    #         out3 = stn(img[:, 2:3],flow)
    #         out = torch.cat([out1, out2, out3], dim=1)
    #         rect_color = out.squeeze().cpu().numpy()
    #         rect_color = F.interpolate(input, size=(h,w), mode='bilinear', align_corners=False)
    #         rect_color = rect_color.transpose(1, 2, 0)
    #         savePath5 = os.path.join(args.imgsave,'{}'.format(timenow), 'rectimg_{}'.format(data))
    #         save_image(rect_img, savePath5)
        
            
            
            
    
    
    
        
    
    
    
    
