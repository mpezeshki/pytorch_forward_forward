'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-27 16:43:08
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-28 09:14:21
FilePath: /pytorch_forward_forward/SplitFF.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''

import torch
from networks.Model import FFNet
from dataloaders.dataset import MNIST_loaders
from utils.misc import overlay_y_on_x
DEVICE = torch.device('cuda')  
torch.manual_seed(1234)
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in trange(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()



class FFNet(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).to(DEVICE)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)



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