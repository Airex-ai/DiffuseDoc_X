import os
import cv2
import time
import torch
import logging
import argparse
import numpy as np
import data as Data
import model as Model
import core.logger as Logger

from math import *
from tqdm import tqdm
from PIL import Image
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.metrics import tensor2im

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

#将8位图转化为24位图
def Turnto24(path):
      files = os.listdir(path)
      for file in files:
          img = Image.open(os.path.join(path,file)).convert('RGB')
          img.save(os.path.join(path,file))
               

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,default='config/train.json',help='JOSN file for configuration')
    parser.add_argument('-p','--phase',type=str,choices=['train','test'],help='Run either train or val')
    parser.add_argument('-gpu','--gpu_ids',type=str,default=None)
    parser.add_argument('-debug','-d',action='store_true')
    # parser.add_argument('--test_dir',type=str,default="./test/crop/")
    
    
    # # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)
    
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    
    #20230314改
    # tensorboard
    tensordir = os.path.join('./tensorboard/','diffusedoclog/{}'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
    results = SummaryWriter(tensordir)
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase != opt['phase']: continue  #opt['phase'] = train
        if opt['phase'] == 'train':
            batchSize = opt['datasets']['train']['batch_size']
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
            training_iters = int(ceil(train_set.datalen / float(batchSize)))
        # elif opt['phase'] == 'test':
        #     test_set = Data.create_dataset(dataset_opt, phase)
        #     test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    
    logger.info(" Initial Model Finished")
    
    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    if opt['phase'] == 'train':
        current_step = diffusion.begin_step
        current_epoch = diffusion.begin_epoch
        n_epoch = opt['train']['n_epoch']
        if opt['path']['resume_state']:
            logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

        while current_epoch < n_epoch:
            current_epoch += 1
            # l_pix_ = 0.0
            # l_sim_ = 0.0
            # l_smt_ = 0.0
            # l_tot_ = 0.0
            for istep, train_data in enumerate(tqdm(train_loader)):
                iter_start_time = time.time()
                current_step += 1

                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if (istep+1) % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    #20230314改
                    results.add_scalar('loss/pix', float(logs['l_pix']), global_step=current_step -1)
                    results.add_scalar('loss/rect', float(logs['l_rect']), global_step=current_step - 1)
                    results.add_scalar('loss/ssim', float(logs['l_ssim']), global_step=current_step - 1)
                    results.add_scalar('loss/tot', float(logs['l_tot']), global_step=current_step - 1)
                    
                    t = (time.time() - iter_start_time) / batchSize
                    visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, t, 'Train')
                    visualizer.plot_current_errors(current_epoch, (istep+1) / float(training_iters), logs)
                    if (istep + 1) % 100 == 0:
                        visuals = diffusion.get_current_visuals_train()
                        # images = []
                        # for label, image_numpy in visuals.items():
                            # images.append(image_numpy.transpose([2, 0, 1]))
                        
                        visualizer.display_current_results(visuals, current_epoch, True)
                        
                   
                

                # validation
                if (istep+1) % opt['train']['val_freq'] == 0:
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')
                    diffusion.test_generation(continuous=False)
                    # diffusion.test_registration(continuous=False)
                    visuals = diffusion.get_current_visuals()
                    visualizer.display_current_results(visuals, current_epoch, True)

                    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')
                    
        

            if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
                logger.info('Saving models and training states.')
                diffusion.save_network(current_epoch, current_step)
                
           
        logger.info('End of training.')
        
    else:
        registTime = []
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        test_dir = './test/crop/'
        print(test_dir)
        for testfile in sorted(os.listdir(test_dir)):
            imp = os.path.join(test_dir, testfile)
            im_ori = cv2.imread(imp).astype(np.float32) / 255.0
            h,w,c = im_ori.shape
            
            im = im_ori[..., ::-1]
            im = cv2.resize(im,(288,288))
            im = im[:, :, ::-1] 
            im = im.astype(float) 
            im = im.transpose(2,0,1)
            im = torch.from_numpy(im).unsqueeze(0).float()
            time1 = time.time()
            diffusion.feed_data(im)
            
            print('Rectification ....')
            diffusion.test_rectification(continuous=True)
            time2 = time.time()

            

            visuals = diffusion.get_current_rectification()
            bm = visuals['BM'] # c,h,w
            
            print(bm.shape)
            lbl = F.upsample(bm.unsqueeze(0),size=(h,w),mode='bilinear') #n,c,h,w
            
            lbl = lbl.permute(0,2,3,1)# n,h,w,c
            out = F.grid_sample(torch.from_numpy(im_ori).permute(2,0,1).unsqueeze(0).float(), lbl, align_corners=True)
            img_geo = ((out[0]*255).permute(1, 2, 0).numpy()).astype(np.uint8)
           
            savePath = os.path.join(result_path, 'Rectification_%s'%(testfile))
            
            cv2.imwrite(savePath,img_geo)

    
    
