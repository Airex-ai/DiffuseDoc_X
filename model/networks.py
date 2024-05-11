# -*- coding = utf-8 -*-
# @Time : 2023/3/27 14:41
# @Author : 熊文菲
# @File : networks.py
# @Software : PyCharm
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import functools
import logging
import torch
# torch.cuda.set_device(0)
# print(torch.cuda.current_device())

import torch.nn as nn
from torch.nn import init
logger = logging.getLogger('base')
####################
# initialize
####################
#参数初始化
def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)

        
def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))
        
        
def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
    
    
def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model 
        
#Generator
def define_G(opt):
    model_opt = opt['model']
    
    from model import diffusion
    from model import unet
    from model.rect_unet import registUnetBlock
    from model.seg import U2NETP
    from GeoTr.GeoTr import GeoTr
    from model.PCN import FlowColumn

    model_seg = U2NETP(3,1)

    model_diffuse = unet.UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['diffusion']['image_size']
    )

    # model_rect = registUnetBlock(model_opt['field']['in_channel'],
    #                        model_opt['field']['encoder_nc'],
    #                        model_opt['field']['decoder_nc'])
    model_rect = GeoTr(num_attn_layers=6)
    # model_rect = FlowColumn()

    
    netG = diffusion.GaussianDiffusion(
        model_seg,
        model_diffuse,
        model_rect,
        channels=model_opt['diffusion']['channels'],
        loss_type='l2',    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'],
        loss_lambda=model_opt['loss_lambda']
    )
   
    if opt['phase'] == 'train':
        load_path = opt['path']['resume_state']
        if load_path is None:
            init_weights(netG.denoise_fn, init_type='orthogonal')
            init_weights(netG.rect_fn, init_type='normal')
            
            
            segdir = './models/seg/seg.pth'
            reload_segmodel(model_seg,segdir)
            # for  p in model_seg.parameters():
            #     p.requires_grad = False
            model_seg.eval()
            # model_seg.train()
    
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG