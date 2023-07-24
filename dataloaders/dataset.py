'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-23 17:15:37
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-24 15:49:01
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
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import utils.misc as misc
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim


# ---------------------------------------------------------------------------- #
#                             Dataloader Generation                            #
# ---------------------------------------------------------------------------- #
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
 
    # ----------------------------------- 导入数据 ----------------------------------- #
    path_audio = path+'audio/audio/'
    data = pd.read_csv(path+'esc50.csv')
    
    train_data = misc.DataGenerator(path_audio, kind='train')
    val_data = misc.DataGenerator(path_audio, kind='val')
    test_data = misc.DataGenerator(path_audio, kind='test')
    
    # ------------------------------- 制作dataloader ------------------------------- #
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, val_loader


if __name__ == "__main__":
    

    train_loader, test_loader,val_loader = ESC50_loaders()
    
    class SimpleCnn(nn.Module):
        def __init__(self):
            super(SimpleCnn, self).__init__()
            self.conv1 = nn.Conv1d(100, 128, kernel_size=5, stride=4)
            self.bn1 = nn.BatchNorm1d(128)
            self.pool1 = nn.MaxPool1d(4)
            self.conv2 = nn.Conv1d(128, 256, 3)
            self.bn2 = nn.BatchNorm1d(256)
            self.pool2 = nn.MaxPool1d(4)
            self.conv3 = nn.Conv1d(256, 512, 3)
            self.bn3 = nn.BatchNorm1d(512)
            self.pool3 = nn.MaxPool1d(4)
            self.conv4 = nn.Conv1d(512, 256, 3)
            self.bn4 = nn.BatchNorm1d(256)
            self.pool4 = nn.MaxPool1d(4)
            self.fc1 = nn.Linear(256, 50)
            
        def forward(self, x):
            x = x.unsqueeze(-1).view(-1, 100, 2205)
            x = self.conv1(x)
            x = F.relu(self.bn1(x))
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(self.bn2(x))
            x = self.pool2(x)
            x = self.conv3(x)
            x = F.relu(self.bn3(x))
            x = self.pool3(x)
            x = self.conv4(x)
            x = F.relu(self.bn4(x))
            x = self.pool4(x)
            x = x.squeeze(-1)
            x = self.fc1(x)
            return x
    
    
    device = torch.device('cuda:2')
    model = SimpleCnn()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=5, device='cuda:2'):
        for epoch in range(epochs):
            training_loss = 0.0
            valid_loss = 0.0
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)
                loss.backward()
                optimizer.step()
                training_loss += loss.data.item()*inputs.size(0)
            training_loss /= len(train_loader.dataset)
            
            model.eval()
            num_correct = 0
            num_examples = 0
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)
                valid_loss += loss.data.item()*inputs.size(0)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)
            
            print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, '
                'accuracy = {:.2f}'.format(epoch+1, training_loss, valid_loss, num_correct/num_examples))

    train(model, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, epochs=30, device=device)
