'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-27 16:43:08
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-28 17:19:55
FilePath: /pytorch_forward_forward/SplitFF.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''

import torch
from networks.Model import FFNet_shallow, FFNet_deep
from dataloaders.dataset import MNIST_loaders
from utils.misc import overlay_y_on_x
DEVICE = torch.device('cuda')  
torch.manual_seed(1234)

# ---------------------------------- 设置网络结构 ---------------------------------- #

net_shallow = FFNet_shallow([784,500]).to(DEVICE) #浅层模型
net_deep = FFNet_deep([500, 500]).to(DEVICE) #深层模型

# ----------------------------------- 读取数据 ----------------------------------- #
train_loader, test_loader = MNIST_loaders()
x, y = next(iter(train_loader[0]))
x, y = x.to(DEVICE), y.to(DEVICE)
x_pos = overlay_y_on_x(x, y)
rnd = torch.randperm(x.size(0))
x_neg = overlay_y_on_x(x, y[rnd])
x_te, y_te = next(iter(test_loader))
x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)

# ----------------------------------- 浅层训练 ----------------------------------- #
out_pos, out_neg = net_shallow.train_in_shallow(x_pos, x_neg)
out_pos, out_neg = out_pos.to(DEVICE), out_neg.to(DEVICE)

# ----------------------------------- 深层训练 ----------------------------------- #
net_deep.train_in_deep(out_pos, out_neg)

# ------------------------------------ 预测 ------------------------------------ #
out_x = net_shallow.predict(x)
out_te = net_shallow.predict(x_te)
print('train error of FF:', 1.0 - net_deep.predict(out_x).eq(y).float().mean().item())
print('test error of FF:', 1.0 - net_deep.predict(out_te).eq(y_te).float().mean().item())
