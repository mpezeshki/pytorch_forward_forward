'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-23 17:15:37
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-28 10:54:50
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
from torch.utils.data import DataLoader, Subset
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import utils.misc as misc
import pandas as pd
import torch


# ---------------------------------------------------------------------------- #
#                                 ESC50_loaders                                #
# ---------------------------------------------------------------------------- #
def ESC50_loaders(path='/home/datasets/SNN/data/ESC50/', batch_size=64, num_subsets=1):
    '''
    num_subsets：训练集划分数量
    description: 用于生成ESC-50的train_loader, test_loader。每个音频样本的有220500点，一共50类
    return {*}: train_loader, test_loader
    '''    
    path_audio = path + 'audio/audio/'
    data = pd.read_csv(path + 'esc50.csv')
    
    train_data = misc.DataGenerator(path_audio, kind='train')
    test_data = misc.DataGenerator(path_audio, kind='test')
    
    # Calculate the size of each subset
    subset_size = len(train_data) // num_subsets
    
    train_loaders = []
    for i in range(num_subsets):
        # Calculate the starting and ending indices for each subset
        start_idx = i * subset_size
        end_idx = start_idx + subset_size
        
        # Create a subset of the train_data using the indices
        subset = Subset(train_data, list(range(start_idx, end_idx)))
        
        # Create a DataLoader for each subset
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return train_loaders, test_loader

# ---------------------------------------------------------------------------- #
#                                      MSD                                     #
# ---------------------------------------------------------------------------- #
def MSD_loaders(path='/home/datasets/SNN/data/MSD/', batch_size=64, num_subsets=1):
    '''
    num_subsets：训练集划分数量
    description: 生成MSD的dataloader, 一共6类
    return: 
    '''
    train_dir = path + 'train'
    test_dir = path + 'test'
    valid_dir = path + 'valid'
    
    image_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
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
    
    # Calculate the size of each subset
    subset_size = len(data['train']) // num_subsets
    
    train_loaders = []
    for i in range(num_subsets):
        # Calculate the starting and ending indices for each subset
        start_idx = i * subset_size
        end_idx = start_idx + subset_size
        
        # Create a subset of the train dataset using the indices
        subset = Subset(data['train'], list(range(start_idx, end_idx)))
        
        # Create a DataLoader for each subset
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    
    valid_loader = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    
    return train_loaders, test_loader, valid_loader

# ---------------------------------------------------------------------------- #
#                                     MNIST                                    #
# ---------------------------------------------------------------------------- #

def MNIST_loaders(batch_size=50000, num_subsets=1):
    '''
    num_subsets：训练集划分数量
    description: 输入batch_size和需要划分的数量
    return {*}
    '''    
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_dataset = MNIST('/home/datasets/SNN/data/', train=True, download=False, transform=transform)
    test_dataset = MNIST('/home/datasets/SNN/data/', train=False, download=False, transform=transform)

    # Calculate the size of each subset
    subset_size = len(train_dataset) // num_subsets

    train_loaders = []
    for i in range(num_subsets):
        # Calculate the starting and ending indices for each subset
        start_idx = i * subset_size
        end_idx = start_idx + subset_size

        # Create a subset of the train_dataset using the indices
        subset = Subset(train_dataset, list(range(start_idx, end_idx)))

        # Create a DataLoader for each subset
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loader

