import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
import time 
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as tortra

from networks.Model import FFAlexNet, BPNet
from dataloaders.dataset import MNIST_loaders
from utils import misc
from utils.generateData import prepare_data



torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


DEVICE = torch.device('cuda:2')
config ={
    'lr' : 0.001,
    'epoch': 50,
    'batchsize': 1000
    
}
writer = SummaryWriter(comment=f"LR_{config['lr']}_EPOCH_{config['epoch']}_rewriteFF")


    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()


def FFAlexNet_experiment():
    transform = tortra.Compose([
            tortra.ToTensor(),
            tortra.Normalize((0.1307,), (0.3081,))
            ])
    train_loader, test_loader = MNIST_loaders(batch_size=1000, transform=transform)

    net = FFAlexNet().to(DEVICE)
    
   
    FF_start_time = time.time()
    train_acc = [] 
    for i, (x, y) in enumerate(train_loader[0]):
    

        x, y = x, y
        # x_pos = x
        # x_neg = x
        x_pos = misc.Conv_overlay_y_on_x(x, y)
        rnd = torch.randperm(x.size(0), device = y.device)
        x_neg = misc.Conv_overlay_y_on_x(x, y[rnd])
        x_pos, x_neg, y = x_pos.to(DEVICE), x_neg.to(DEVICE), y.to(DEVICE)
    
        for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
            visualize_sample(data, name)
    
        net.ftrain(x_pos, x_neg)
        acc = net.predict(x).eq(y).float().mean().item()
        train_acc.append(acc)
        print(f"epoch{i}: test acc, {acc}")
        writer.add_scalar('FFAlexnetAccuracy/trainacc',acc, i)
        
    print(f'Epoch {i} train acc of FF:', sum(train_acc)/len(train_acc))
    FF_end_time = time.time()
    journey = FF_end_time - FF_start_time
    writer.add_scalar('FFAlexnetAccuracy/train',sum(train_acc)/len(train_acc))
    writer.add_scalar('Time/FFAlexnettime', journey)  
    
    
    acc_list = []
    for x_te, y_te in test_loader:

        x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)
        acc = net.predict(x_te).eq(y_te).float().mean().item() 
        acc_list.append(acc)
        break    
        
    print('test acc of FF:', sum(acc_list)/len(acc_list))
    
    writer.add_scalar('FFAlexnetAccuracy/test', sum(acc_list)/len(acc_list))

    #print('test acc of FF:', net.predict(x_te).eq(y_te).float().mean().item())


def BP_experiment():
    BPtrain_loader, BPtest_loader = MNIST_loaders(512, 512)

    BP_net = BPNet([784, 500, 10]).to(DEVICE)
    BP_loss = nn.CrossEntropyLoss(reduction='none')
    BP_optim = torch.optim.Adam(BP_net.parameters(), lr = config['lr'])
    bestacc = 0.0
    minloss = 1.0
    journeylist = []
    for epoch in range(config['epoch']):
        BP_start_time = time.time()
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
        BP_end_time = time.time()
        journeylist.append(BP_end_time-BP_start_time)
        
        
        avg_loss, avg_acc = BP_metric[0] / BP_metric[2], BP_metric[1] / BP_metric[2]
        writer.add_scalar('BPLoss/train', avg_loss, epoch)
        writer.add_scalar('BPAccuracy/train', avg_acc, epoch)
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
        writer.add_scalar('BPAccuracy/test', testacc, epoch)
    
    writer.add_scalar('Time/BPtime', sum(journeylist))

    print(f"BP trainacc:{bestacc} \n BP trainloss:{minloss}")
    print(f"BP testacc:{testacc}\n") 
 

#############Travis_Python_Annotation#############
# Try to use hybrid image to train.
#
#
#
##################################################
def FFAlexNetHybrid_experiment():
    
    ## prepare data
    from torch.utils.data import DataLoader
    # prepare_data()
    
    transform = tortra.Compose([
            tortra.ToTensor(),
            tortra.Normalize((0.1307,), (0.3081,))
            ])

    pos_dataset = MNIST(root='./data', download=False, transform=transform, train=True)
    pos_dataloader = DataLoader(pos_dataset, batch_size=config['batchsize'], shuffle=True)
    neg_dataset = torch.load('./data/transformed_dataset.pt')
    neg_dataloader = DataLoader(neg_dataset, batch_size=config['batchsize'], shuffle=True, num_workers=4)
    
    # Load the test images
    test_dataset = MNIST(root='./data', train=False, download=False, transform=transform)
    # Create the data loader
    test_dataloader = DataLoader(test_dataset, batch_size=config['batchsize'], shuffle=True, num_workers=4)
    
    ## Begin to train 
    
    
    net = FFAlexNet().to(DEVICE)
    
    
    FF_start_time = time.time()
    train_acc = [] 
    epochtqdm = tqdm(range(config['epoch']), desc= "Global Iteration==>", position=0)
   
    for epoch in epochtqdm:
        datatqdm = tqdm(zip(pos_dataloader, neg_dataloader), desc= f"Local Iteration {epoch}", leave=False, position=1)
        acclist = []
        # meter = misc.Accumulator(1)
        for pos_data, neg_imgs in datatqdm:
            x_pos, y = pos_data
            x_neg = neg_imgs.unsqueeze(1)
            x_pos = x_pos.to(DEVICE)
            x_neg = x_neg.to(DEVICE)
            y = y.to(DEVICE)
            
            net.ftrain(x_pos, x_neg)
   
            acc = net.predict(x_pos).eq(y).float().mean().item()
            acclist.append(acc)
            print(f"acc:{acc}\n")
            # meter.add(acc)
        a = sum(acclist) /len(acclist)   
        print(f"epoch{epoch}: train acc, {a}")
            
   
    # for i, (x, y) in enumerate(train_loader[0]):
    

    #     x, y = x, y
    #     # x_pos = x
    #     # x_neg = x
    #     x_pos = misc.Conv_overlay_y_on_x(x, y)
    #     rnd = torch.randperm(x.size(0), device = y.device)
    #     x_neg = misc.Conv_overlay_y_on_x(x, y[rnd])
    #     x_pos, x_neg, y = x_pos.to(DEVICE), x_neg.to(DEVICE), y.to(DEVICE)
    
    #     for data, name in zip([x, x_pos, x_neg], ['orig', 'pos', 'neg']):
    #         visualize_sample(data, name)
    
    #     net.ftrain(x_pos, x_neg)
    #     acc = net.predict(x).eq(y).float().mean().item()
    #     train_acc.append(acc)
    #     print(f"epoch{i}: test acc, {acc}")
    #     writer.add_scalar('FFAlexnetAccuracy/trainacc',acc, i)
        
    # print(f'Epoch {i} train acc of FF:', sum(train_acc)/len(train_acc))
    # FF_end_time = time.time()
    # journey = FF_end_time - FF_start_time
    # writer.add_scalar('FFAlexnetAccuracy/train',sum(train_acc)/len(train_acc))
    # writer.add_scalar('Time/FFAlexnettime', journey)  
    
    
    # acc_list = []
    # for x_te, y_te in test_loader:

    #     x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)
    #     acc = net.predict(x_te).eq(y_te).float().mean().item() 
    #     acc_list.append(acc)
    #     break    
        
    # print('test acc of FF:', sum(acc_list)/len(acc_list))
    
    # writer.add_scalar('FFAlexnetAccuracy/test', sum(acc_list)/len(acc_list))







    
if __name__ == "__main__":
    torch.manual_seed(1234)

    # FFAlexNet_experiment()

    ### BP
    #BP_experiment()
    FFAlexNetHybrid_experiment()
    print(f"Done")

