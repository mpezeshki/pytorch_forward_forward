'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-23 17:15:37
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-25 10:33:30
FilePath: /pytorch_forward_forward/dataloaders/dataset.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''
'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-23 17:15:37
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-24 13:55:11
FilePath: /pytorch_forward_forward/dataloaders/dataset.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''
from torchvision import datasets
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda, transforms
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import utils.misc as misc
import pandas as pd
import torch




def MNIST_loaders(train_batch_size=50000, test_batch_size=10000): 
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])
    ## transform处就展平了

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=False,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=False,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

def ESC50_loaders(path='./data/ESC50/', batch_size = 64):
    '''
    description: 用于生成ESC-50的train_loader, test_loader。每个音频样本的有220500点,一共50类

    return {*}: train_loader, test_loader
    '''    
    path_audio = path+'audio/audio/'
    data = pd.read_csv(path+'esc50.csv')
    
    train_data = misc.DataGenerator(path_audio, kind='train')
    # val_data = misc.DataGenerator(path_audio, kind='val')
    test_data = misc.DataGenerator(path_audio, kind='test')
    
    # ------------------------------- 制作dataloader ------------------------------- #
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def MSD_loaders(path='./data/MSD/', batch_size = 64):
    '''
    description: 生成MSD的dataloader, 一共6类
    return: 
    '''    
    train_dir = path + 'train'
    test_dir = path + 'test'
    valid_dir = path + 'valid'
    CLASS_NAMES = os.listdir(train_dir)
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            
        ]),
        
        'valid': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }
    data = {
    'train': datasets.ImageFolder(root=train_dir, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_dir, transform=image_transforms['valid'])
    }
    train_loader = DataLoader(data['train'], batch_size = batch_size, shuffle=True)
    valid_loader = DataLoader(data['valid'], batch_size = batch_size, shuffle=True)
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    return train_loader, test_loader, valid_loader



