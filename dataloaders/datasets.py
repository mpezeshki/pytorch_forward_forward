'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-23 17:15:37
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-24 09:41:04
FilePath: /pytorch_forward_forward/datasets.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader



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


if __name__ == "__main__":
    MNIST_loaders()