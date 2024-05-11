import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
from model import metrics as Metrics
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))

        self.schedule_phase = None
        self.centered = opt['datasets']['centered']

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
        self.load_network()
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.5, 0.999))
            self.log_dict = OrderedDict()
            # self.schedule_opt = ReduceLROnPlateau(self.optG,'min',factor=0.5, patience=5, verbose=True)
        self.print_network(self.netG)

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        score, loss = self.netG(self.data)
        self.score, self.out_M, self.bm = score
        # need to average in multi-gpu

        # l_tot = loss
        # l_pix, l_sim, l_smt, l_tot = loss
        l_pix, l_rect,l_ssim, l_tot = loss
        

        l_tot.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        # self.log_dict['l_sim'] = l_sim.item()
        # self.log_dict['l_smt'] = l_smt.item()
        self.log_dict['l_rect'] = l_rect.item()
        self.log_dict['l_ssim'] = l_ssim.item()
        self.log_dict['l_tot'] = l_tot.item()
        
        # del l_pix, l_smt, l_sim, l_tot

    def test_generation(self, continuous=False):
        self.netG.eval()
        if self.opt['phase'] == 'train':
            input = torch.cat([self.data['M'], self.data['F']], dim=1)
            # input = self.data['M']
            if isinstance(self.netG, nn.DataParallel):
                self.MF, self.out_M = self.netG.module.generation(input, continuous)
            else:
                self.MF, self.out_M= self.netG.generation(input, continuous)
            self.netG.train()
        else:
            input = self.data
            if isinstance(self.netG, nn.DataParallel):
                self.MF, self.out_M = self.netG.module.generation(input, continuous)
            else:
                self.MF, self.out_M= self.netG.generation(input, continuous)

    def test_rectification(self, continuous=False):
        self.netG.eval()
        if self.opt['phase'] == 'train':
            input = torch.cat([self.data['M'], self.data['F']], dim=1)
            if isinstance(self.netG, nn.DataParallel):
                self.out_M, self.bm = self.netG.module.rectification(input)
            else:
                self.out_M, self.bm = self.netG.rectification(input)
            self.netG.train()
        else:
            input = self.data
            if isinstance(self.netG, nn.DataParallel):
                self.out_M, self.bm = self.netG.module.rectification(input)
            else:
                self.out_M, self.bm = self.netG.rectification(input)

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals_train(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        out_dict['M'] = Metrics.tensor2im(self.data['M'][0].detach().float().cpu(), min_max=(0, 1))
        out_dict['F'] = Metrics.tensor2im(self.data['F'][0].detach().float().cpu(), min_max=(0, 1))
        out_dict['out_M'] = Metrics.tensor2im(self.out_M[0].detach().float().cpu(), min_max=(0, 1))
        # out_dict['out_M'] = Metrics.tensor2im(self.out_M.detach().float().cpu(), min_max=min_max)
        out_dict['BM'] = Metrics.tensor2im(self.bm[0].detach().float().cpu(), min_max=(0, 1))
        
        return out_dict

    def get_current_visuals(self, sample=False):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        out_dict['MF'] = Metrics.tensor2im(self.MF[0].detach().float().cpu(), min_max=(0, 1))
        out_dict['M'] = Metrics.tensor2im(self.data['M'][0].detach().float().cpu(), min_max=(0, 1))
        out_dict['F'] = Metrics.tensor2im(self.data['F'][0].detach().float().cpu(), min_max=(0, 1))
        out_dict['out_M'] = Metrics.tensor2im(self.out_M[0].detach().float().cpu(), min_max=(0, 1))
        # out_dict['out_M'] = Metrics.tensor2im(self.out_M.detach().float().cpu(), min_max=min_max)
        out_dict['BM'] = Metrics.tensor2im(self.bm[0].detach().float().cpu(), min_max=(0, 1))
        return out_dict

    def get_current_generation(self):
        out_dict = OrderedDict()
        # out_dict['MF'] = Metrics.tensor2im(self.MF[0].detach().float().cpu(), min_max=(0, 1))
        # out_dict['MF'] = self.MF[0].detach().float().cpu()
        out_dict['MF'] = self.MF[0]

        # out_dict['out_M'] = self.out_M[0].detach().float().cpu()
        out_dict['out_M'] = self.out_M[0]
        # out_dict['out_M'] = Metrics.tensor2im(self.out_M[0].detach().float().cpu(), min_max=(0, 1))
        return out_dict

    def get_current_rectification(self):
        out_dict = OrderedDict()
        out_dict['out_M'] =self.out_M[0].detach().float().cpu()
        out_dict['BM'] = self.bm[0].detach().float().cpu()
        return out_dict

    def print_network(self, net):
        s, n = self.get_network_description(net)
        if isinstance(net, nn.DataParallel):
            net_struc_str = '{} - {}'.format(net.__class__.__name__,
                                             net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(net.__class__.__name__)

        logger.info(
            'Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        genG_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen_G.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        
        # rect_path = os.path.join(
        #     self.opt['path']['checkpoint'], 'I{}_E{}_rect.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, genG_path)
        
        # rect_dict = self.netG.model_rect.state_dict()
        # for key, param in state_dict.items():
        #     rect_dict[key] = param.cpu()
        # torch.save(rect_dict, rect_path)

        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(genG_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']

        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            genG_path = '{}I46033_E1_gen_G.pth'.format(load_path)

            opt_path = '{}I46033_E1_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                genG_path), strict=(not self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                # self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']