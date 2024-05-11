import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

from skimage import io
from model.seg import U2NETP
from torch.autograd import Variable
from model.metrics import tensor2im
from model.networks import init_weights

from tensorboardX import SummaryWriter


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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


seg_loss = nn.BCELoss(size_average=True)


def Loss(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = seg_loss(d0, labels_v)
    loss1 = seg_loss(d1, labels_v)
    loss2 = seg_loss(d2, labels_v)
    loss3 = seg_loss(d3, labels_v)
    loss4 = seg_loss(d4, labels_v)
    loss5 = seg_loss(d5, labels_v)
    loss6 = seg_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapathTrain', type=str, default='/media/wit/HDD_0/xwf/data/seg/train/')
    parser.add_argument('--datapathTest', type=str, default='/media/wit/HDD_0/xwf/DiffuseDoc/Documet/seg/test/')
    parser.add_argument('--epochs', type=int, default=40, metavar='N')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LN')
    parser.add_argument('--batchsize', type=int, default=2, metavar='N')
    parser.add_argument('--savemodelPath', type=str, default='./models/segmodel/')
    parser.add_argument('--use_shuffle', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8, metavar='N')
    parser.add_argument('--parallel', default=False, type=str2bool, help='Set to True to train on parallel GPUs')
    parser.add_argument('--beta1', default=0.9, help='Beta Values for Adam Optimizer')
    parser.add_argument('--beta2', default=0.999, help='Beta Values for Adam Optimizer')
    parser.add_argument('--logroot', type=str, default='./logs/seglog/')
    parser.add_argument('--imgsave', type=str, default='./images/')
    args = parser.parse_args()

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    # log
    tensorboader_dir = os.path.join(args.logroot, 'tensorboder/{}'.format(time.strftime("%Y-%m-%d %H:%M:%S")))
    if not os.path.exists(tensorboader_dir):
        os.makedirs(tensorboader_dir)

    results = SummaryWriter(tensorboader_dir)

    # Logger.setup_logger(None,args.logroot,'seg',level=logging.INFO)
    # logger = logging.getLogger('base')

    # dataset
    from data.SEGDataset import SEGData as S

    train_dataset = S(dataroot=args.datapathTrain)
    test_dataset = S(dataroot=args.datapathTest)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize,
                                               shuffle=args.use_shuffle, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize,
                                              shuffle=args.use_shuffle, num_workers=args.num_workers)

    # model
    # model = U2NETP(3, 1)#彩色图像
    model = U2NETP(1, 1)#灰度图像
    init_weights(model, init_type='kaiming')
    if args.parallel:
        model = torch.nn.DataParallel(model)
    else:
        model = model.cuda()

    # set
    # seg_loss = nn.CrossEntropyLoss()
    # seg_loss = nn.BCELoss(size_average=True)

    optimizer = optim.Adam(model.parameters(),
                           lr=float(args.lr),
                           betas=(float(args.beta1), float(args.beta2)))

    timenow = '{}'.format(time.strftime("%Y-%m-%d %H:%M:%S"))
    modeldir = args.savemodelPath + '{}/'.format(timenow)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    # ite_num = 0
    # running_loss = 0.0
    # running_tar_loss = 0.0
    # ite_num4val = 0
    # train
    for epoch in range(args.epochs):
        # ite_num = ite_num + 1
        # ite_num4val = ite_num4val + 1
        if isinstance(model, torch.nn.Module):
            model.train()
        for step, train_data in enumerate(tqdm(train_loader)):
            train_data = feed_data_to_device(train_data)
            d0, d1, d2, d3, d4, d5, d6 = model(train_data['data'])
            # print(train_data['label']) # (2,1,256,256)
            # print(train_data['label'].shape) # (2,1,256,256)

            _, loss = Loss(d0, d1, d2, d3, d4, d5, d6, train_data['label'])

            iters = epoch * len(train_loader) + step

            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            # running_loss += loss.data.item()
            # running_tar_loss += loss_2.data.item()

            # del d0, d1, d2, d3, d4, d5, d6, loss0, loss1,loss2, loss3, loss4, loss5, loss6, loss, loss_2

            results.add_scalar('loss/train', float(loss.data), global_step=iters)

            # print("epoch: [{}/{}][{}/{}] \t seg_loss: {}\n".format(epoch+1, args.epochs, iters, len(train_loader), loss.data))
            # validation
            if ((step + 1) % 6000 == 0):
                print("epoch: [{}/{}][{}/{}] \t seg_loss: {}\n".format(epoch + 1, args.epochs, iters, len(train_loader),
                                                                       loss.data))
                print("\nvalidating ...")
                with torch.no_grad():
                    for step1, test_data in enumerate(tqdm(test_loader)):
                        test_data = feed_data_to_device(test_data)
                        d0, d1, d2, d3, d4, d5, d6 = model(test_data['data'])
                        _, loss_test = Loss(d0, d1, d2, d3, d4, d5, d6, test_data['label'])

                        iters1 = epoch * len(test_loader) + step1

                        # save validation image
                        # if(step1 % 800 == 0):
                        #     pass

                        results.add_scalar('loss/test', float(loss_test.data), global_step=iters1)

        modelpath = modeldir + '{}seg.pth'.format(epoch)
        print('saving model......')
        torch.save(model.state_dict(), modelpath)
        print('done..............')

        # save model
    modelpath = modeldir + 'seg.pth'
    print('saving model......')
    torch.save(model.state_dict(), modelpath)
    print('all done..............')

    results.close()





