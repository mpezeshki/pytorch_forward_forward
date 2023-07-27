'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-27 16:43:08
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-27 21:44:08
FilePath: /pytorch_forward_forward/SplitFF.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''


import torch
from networks.Model import FFNet
from dataloaders.dataset import MNIST_loaders
from utils import misc
from utils.misc import *

DEVICE = torch.device('cuda')  
torch.manual_seed(1234)
train_loader, test_loader = MNIST_loaders()

net = FFNet([784, 500, 500]).to(DEVICE)
x, y = next(iter(train_loader[0]))
x, y = x.to(DEVICE), y.to(DEVICE)
x_pos = overlay_y_on_x(x, y)
rnd = torch.randperm(x.size(0))
x_neg = overlay_y_on_x(x, y[rnd])

net.train(x_pos, x_neg)

print('train error of FF:', 1.0 - net.predict(x).eq(y).float().mean().item())

x_te, y_te = next(iter(test_loader))
x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)

print('test error of BP:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())