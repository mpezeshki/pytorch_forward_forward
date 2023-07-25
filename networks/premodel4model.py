'''
Author: linjunnuo limchvnno@gmail.com
Date: 2023-07-25 10:17:07
LastEditors: linjunnuo limchvnno@gmail.com
LastEditTime: 2023-07-25 10:20:59
FilePath: /pytorch_forward_forward/networks/premodel4model.py
Description: 

Copyright (c) 2023 by linjunnuo , All Rights Reserved. 
'''
import torch.nn as nn
from torchsummary import summary
from torchvision import models
import warnings
warnings.filterwarnings("ignore")

def MSC_premodel():
    '''
    description: 用于MSC的模型

    '''    
    train_loader, valid_loader = MSD_loader()
    device = torch.device("cuda:2")
    model = models.resnet50(pretrained = True).to(device)
    summary(model, (3, 224, 224))
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, 6)
    )

    # Transfer the model to device
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 20
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        train_loss = 0
        valid_loss = 0

        # Training loop
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
        
            # Validation loop
        model.eval()
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * images.size(0)
            valid_loss /= len(valid_loader.dataset)
            valid_losses.append(valid_loss)
        
        # Print training and validation loss
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))


def ESC_premodel():
    '''
    description: 用于ESC的模型
    '''    
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