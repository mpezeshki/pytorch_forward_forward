'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-27 16:43:08
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-28 20:56:16
FilePath: /pytorch_forward_forward/SplitFF.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''

import torch
import time
from networks.Model import FFNet_shallow, FFNet_deep
from dataloaders.dataset import MNIST_loaders
from utils.misc import overlay_y_on_x
DEVICE = torch.device('cuda')  
from torch.utils.tensorboard import SummaryWriter

config = {
    'lr': 0.001,
    'epoch': 50,
}
writer = SummaryWriter(
    comment=f"LR_{config['lr']}_EPOCH_{config['epoch']}_rewriteFF")


# ---------------------------------- 设置网络结构 ---------------------------------- #

net_shallow = FFNet_shallow([784,500]).to(DEVICE) #浅层模型
net_deep = FFNet_deep([500, 500]).to(DEVICE) #深层模型

# ----------------------------------- 读取数据 ----------------------------------- #
train_loader, test_loader = MNIST_loaders()
x, y = next(iter(train_loader[0]))
# ----------------------------------- 开始训练 ----------------------------------- #
FF_start_time = time.time()
train_acc = []
for i, (x, y) in enumerate(train_loader[0]):

    x, y = x.to(DEVICE), y.to(DEVICE)
    x_pos = overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = overlay_y_on_x(x, y[rnd])
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)
    out_pos, out_neg = net_shallow.train_in_shallow(x_pos, x_neg)
    out_pos, out_neg = out_pos.to(DEVICE), out_neg.to(DEVICE)
    net_deep.train_in_deep(out_pos, out_neg)
    out_x = net_shallow.predict(x)
    train_acc.append(net_deep.predict(out_x).eq(y).float().mean().item())
    break
print(f'Epoch {i} train acc of FF:', sum(train_acc)/len(train_acc))
FF_end_time = time.time()
journey = FF_end_time - FF_start_time
writer.add_scalar('FFAccuracy/train', sum(train_acc)/len(train_acc))
writer.add_scalar('Time/FFtime', journey)

acc_list = []
for x_te, y_te in test_loader:
    x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)
    out_x_te = net_shallow.predict(x_te)
    acc = net_deep.predict(out_x_te).eq(y_te).float().mean().item()
    acc_list.append(acc)
    break

print('test acc of FF:', sum(acc_list)/len(acc_list))
writer.add_scalar('FFAccuracy/test', sum(acc_list)/len(acc_list))


