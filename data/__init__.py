# -*- coding = utf-8 -*-
# @Time : 2023/3/27 14:24
# @Author : 熊文菲
# @File : __init__.py.py
# @Software : PyCharm
# -*- coding = utf-8 -*-
# @Time : 2023/2/20 14:05
# @Author : 熊文菲
# @File : __init__.py.py
# @Software : PyCharm
'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'test':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))

def create_dataset(dataset_opt, phase):
    '''create dataset'''
    #加载wropdoc数据集
    # from data.DOCDataset import DocData as D
    # dataset = D(dataroot=dataset_opt['dataroot'],
    #             split=phase
    #             )
    # dataset = D(dataroot=dataset_opt,
    #             split=phase
    #             )
    #加载docproj数据集
    # from data.DocProjDataset import DocProjDataset as D
    # dataset = D(dataroot=dataset_opt['dataroot'],
    #             split=phase
    #             )
    #加载Doc3D数据集
    # from data.Doc3D import Doc3D as D
    # dataset = D(dataroot=dataset_opt['dataroot'],split=phase,
    #             is_transform=dataset_opt['is_transform'])
    #加载mydata
    from data.MyData import MyData as D
    
    dataset = D(dataroot=dataset_opt['dataroot'],split=phase)
    
    return dataset