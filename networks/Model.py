from typing import Union
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.optim import Adam
from torchvision import models
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from tqdm import trange

from utils.misc import overlay_y_on_x

DEVICE = torch.device('cuda')
writer = SummaryWriter(comment=f"FFLayer")

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
        # for i in tqdm(range(self.num_epochs)):
        for i in trange(self.num_epochs):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            writer.add_scalar("FFLoss/Layer", loss, i)
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
        # goodness_per_label [50000, 10]
        # goodness_per_label.argmax(1) [50000, 1] 
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class BPNet(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []

        self.model = nn.Sequential()
        for d in range(len(dims) - 1):
            self.model.append(nn.Linear(dims[d], dims[d + 1]))
            self.model.append(nn.ReLU())


    def forward(self, x):
        y = self.model(x)
        return y

    # def trainnet(self, x):
    #     y = self.forward(x)
    #     return y

    def predict(self, x):
        y = self.forward(x)

        return y

class ConvLayer(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride= 1, padding = 0, dilation= 1, groups= 1, bias= True, padding_mode= 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 10000 
    
    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            writer.add_scalar("FFLoss/Layer", loss, i)
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()
        


class FFAlexNet(torch.nn.Module):
    def __init__(self, dims, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = [
            ConvLayer(1, 96, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvLayer(96, 256,kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvLayer(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvLayer(384, 384),
            nn.ReLU(),
            ConvLayer(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            Layer(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            Layer(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            Layer(4096, 10)
        ]
         
        
        
    
    def forward(self, x):
        y = self.model(x)
        return y
    
    def predict(self, x):
        y = self.forward(x)
        
        return y 

class FFNet_shallow(torch.nn.Module):
    '''
    description: 浅层模型
    param {*} h_pos, h_neg: 浅层模型最后输出的一层特征,直接输入给深层网络
    param {*} result: _out浅层模型最后输出的一层特征,直接输入给深层网络
               goodness_per_label:浅层模型的预测精度     
    return {*}
    '''
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).to(DEVICE)]


    def predict(self, x):
        goodness_per_label = []
        _out = torch.zeros(10, x.shape[0], 500).to(DEVICE)
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
            
            _out[label] = h
        goodness_per_label = torch.cat(goodness_per_label, 1)
        result = [_out, goodness_per_label]
        return result
    def train_in_shallow(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer in shallow', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)
        return h_pos, h_neg

class FFNet_deep(torch.nn.Module):
    '''
    description: 深层模型
    param {*} self
    param {*} x
    return {*}
    '''
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).to(DEVICE)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = x[0][label]
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        goodness_per_label_mean = (goodness_per_label + x[1]) / 2.0
        return goodness_per_label_mean.argmax(1)

    def train_in_deep(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer in deep', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)

