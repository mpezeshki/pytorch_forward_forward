from typing import Union
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.optim import Adam
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from tqdm import trange
from utils.misc import overlay_y_on_x
DEVICE = torch.device('cuda')
from tqdm import trange, tqdm

from utils.misc import overlay_y_on_x, Conv_overlay_y_on_x
from networks.block import FC_block


DEVICE = torch.device('cuda:2')
writer = SummaryWriter(comment=f"FFLayer")

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 10

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def ftrain(self, x_pos, x_neg):
        # for i in tqdm(range(self.num_epochs)):
        for i in range(self.num_epochs):
        # for i in trange(self.num_epochs):
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

    def ftrain(self, x_pos, x_neg):
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
    def __init__(self, in_channels, out_channels, kernel_size, stride= 1, padding = 0, dilation= 1, groups= 1, bias= True, padding_mode= 'zeros', device=None, dtype=None, isrelu=True, ismaxpool=True) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.relu = torch.nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.opt = Adam(self.parameters(), lr=0.05)
        self.threshold = 2.0
        self.num_epochs = 1000 
        self.ismaxpool = ismaxpool
        self.isrelu = isrelu

        
        
    def add_compute(self, func):
        pass
        
    def forward(self, input):
        
        output = self._conv_forward(input, self.weight, self.bias)
        if self.isrelu:
            output = self.relu(output)
        if self.ismaxpool:
            output = self.maxpool(output)
        return output
    
    def ftrain(self, x_pos, x_neg):
        
        for i in range(self.num_epochs):
        # for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            T = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean(dim=(1,2))
            mask = torch.isinf(T)
            
            T = T[~mask]
            loss = T.mean()
            writer.add_scalar("FFLoss/Layer", loss, i)
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

class FlattenLayer(nn.Flatten):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__(start_dim, end_dim)
    
    def ftrain(self, x_pos, x_neg):

        return x_pos.flatten(self.start_dim, self.end_dim).detach(), x_neg.flatten(self.start_dim, self.end_dim).detach()      


class FFAlexNet(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = [
            ConvLayer(1, 32, kernel_size=3, padding=1).to(DEVICE),
            ConvLayer(32, 64, kernel_size=3, stride=1, padding=1).to(DEVICE),
            ConvLayer(64, 128, kernel_size=3, padding=1, ismaxpool=False).to(DEVICE),
            ConvLayer(128, 256, kernel_size=3, padding =1, ismaxpool=False).to(DEVICE),
            FlattenLayer().to(DEVICE),
            Layer(12544, 1024).to(DEVICE),
            Layer(1024, 512).to(DEVICE), 
            Layer(512, 10).to(DEVICE)
        ]
         
    def ftrain(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            layer.train()
            h_pos, h_neg = layer.ftrain(h_pos, h_neg)        

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = Conv_overlay_y_on_x(x, label)
            goodness = []
            h = h.to(DEVICE)
            for layer in self.layers:
                h = layer(h)
                if isinstance(layer, FlattenLayer):
                    pass
                else:
                    goodness += [h.pow(2).mean(dim=[di for di in range(1, len(h.shape))])]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        # goodness_per_label [50000, 10]
        # goodness_per_label.argmax(1) [50000, 1] 
        return goodness_per_label.argmax(1)
    
    
hyperparams= [16, 1728, 3, 0.005, 20, 'vehicleimage', 0.001, 'C1']


class SNN(nn.Module):
    def __init__(self, layersize, batchsize):
        """
        layersize is the size of each layer like [24*24*3, 500, 3]
        stepsize is the batchsize.
        """
        super(SNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers_size = layersize
        self.lastlayer_size = layersize[-1]
        self.len = len(self.layers_size) - 1
        self.error = None
        self.stepsize = batchsize
        self.time_windows = 20

        for i in range(self.len):
            self.layers.append(FC_block(self.stepsize,self.time_windows, self.lastlayer_size, self.layers_size[i], self.layers_size[i + 1]))

    def forward(self, input):
        for step in range(self.stepsize):

            x = input > torch.rand(input.size()).to(DEVICE)

            x = x.float().to(DEVICE)
            x = x.view(self.stepsize, -1)
            y = x
            for i in range(self.len):
                y = self.layers[i](y)
#        print('x',x)
#        print('x.shape',x.shape)
        outputs = self.layers[-1].sumspike / self.time_windows 

        return outputs


    


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

class BPNet_split(torch.nn.Module):
    '''
    description: 
    param {*} self
    param {*} dims
    return {*}
    '''    
    def __init__(self, dims):
        super().__init__()
        self.shallow_model = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU()
        )
        self.deep_model = nn.Sequential()
        for d in range(1, len(dims) - 1):
            self.deep_model.append(nn.Linear(dims[d], dims[d + 1]))
            self.deep_model.append(nn.ReLU())

    def forward(self, x):
        shallow_output = self.shallow_model(x)
        deep_output = self.deep_model(shallow_output)
        return deep_output
    
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