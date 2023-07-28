'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-24 13:22:11
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-28 09:58:03
FilePath: /pytorch_forward_forward/FFvsBP.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import torch.nn.functional as F
DEVICE = torch.device('cuda')
EPOCHS = 100

from networks.Model import FFNet, BPNet
from dataloaders.dataset import MNIST_loaders
from utils import misc





    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    net = FFNet([784, 500, 500]).to(DEVICE)
    x, y = next(iter(train_loader[0])
    x, y = x.to(DEVICE), y.to(DEVICE)
    x_pos = misc.overlay_y_on_x(x, y)
    rnd = torch.randperm(x.size(0))
    x_neg = misc.overlay_y_on_x(x, y[rnd])
    
    for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
        visualize_sample(data, name)
    
    net.train(x_pos, x_neg)

    print('train error of FF:', 1.0 - net.predict(x).eq(y).float().mean().item())

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)

    print('test error of BP:', 1.0 - net.predict(x_te).eq(y_te).float().mean().item())


    ### BP
    BPtrain_loader, BPtest_loader = MNIST_loaders(256, 256)

    BP_net = BPNet([784, 500, 10]).to(DEVICE)
    BP_loss = nn.CrossEntropyLoss(reduction='none')
    BP_optim = torch.optim.Adam(BP_net.parameters(), lr = 0.001)
    bestacc = 0.0
    minloss = 1.0
    for epoch in range(EPOCHS):
        BP_metric = misc.Accumulator(3)
        for x, y in tqdm(BPtrain_loader):
            if isinstance(BP_net, torch.nn.Module):
                BP_net.train()
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = BP_net(x)
            loss = BP_loss(y_hat, y)
            BP_optim.zero_grad()
            loss.mean().backward()
            BP_optim.step()

            BP_metric.add(float(loss.sum()), misc.accuracy(y_hat, y), y.numel())
        avg_loss, avg_acc = BP_metric[0] / BP_metric[2], BP_metric[1] / BP_metric[2]
        print(f"Epoch {epoch}: loss {avg_loss}, acc {avg_acc}\n")
        if avg_loss <= minloss:
            minloss = avg_loss
        if avg_acc >= bestacc:
            bestacc = avg_acc

    BP_net.eval()
    testmetric = misc.Accumulator(2)
    with torch.no_grad():
        for X, y in BPtest_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            testmetric.add(misc.accuracy(BP_net(X), y), y.numel())

    testacc = testmetric[0] / testmetric[1]

    print(f"BP trainacc:{bestacc} \n BP trainloss:{minloss}")
    print(f"BP testacc:{testacc}\n")

    print(f"Done")

