import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as tortra


from networks.Model import FFNet, BPNet, SNN
from dataloaders.dataset import MNIST_loaders, debug_loaders
from utils import misc
from utils import federated

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


DEVICE = torch.device('cuda:2')
config = {
    'lr': 0.01,
    'epoch': 10,
    'globalepoch': 200,
    'batchsize': 512
}



def FederatedSNN_experiment(num_of_clients):

    num_of_clients = num_of_clients
    
    transform = tortra.Compose([
            tortra.ToTensor(),
            tortra.Normalize((0.1307,), (0.3081,))
            ])
    train_loader, test_loader = MNIST_loaders(batch_size=config['batchsize'], transform=transform, num_subsets=num_of_clients)
    client_step = [iter(_) for _ in train_loader]
    
    server_model = SNN([28*28*1, 500, 10],config['batchsize']).to(DEVICE) 
    client_models = [copy.copy(server_model) for _ in range(num_of_clients)]
    client_optims = [torch.optim.SGD(client_models[i].parameters(), lr = config['lr']) for i in range(len(client_models))]
    loss = nn.CrossEntropyLoss()
    
    
    FF_start_time = time.time()
    train_acc = []
    epoch = 0
    
    outertqdm = tqdm(range(config['globalepoch']), desc=f"Global Epoch", position=0)
    
    for epoch in outertqdm:
        print(f"Global epoch {epoch+1}")
        inertqdm = tqdm(train_loader, desc=f"Local client", position=1, leave=False)
        
        local_avg_acc = []
        for i, iterator in enumerate(inertqdm):
            client_models[i].train()
            metric = misc.Accumulator(3)

            train_acc_list = []
            
            for data in iterator:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)
                client_optims[i].zero_grad()
                y_hat = client_models[i](x)
                l = loss(y_hat, y)
                l.backward()
                client_optims[i].step()
                
                with torch.no_grad():
                    metric.add(l * x.shape[0], misc.accuracy(y_hat, y), x.shape[0])
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                train_acc_list.append(train_acc)
            local_avg_acc.append(sum(train_acc_list)/len(train_acc_list))
            print(f"client {i}: trainacc {sum(train_acc_list)/len(train_acc_list)}")
        
        
        writer.add_scalars('FederatedSNN/localtrainacc',{f"client{i}": j for i, j in enumerate(local_avg_acc)} , epoch)              

        
        client_models, server_model = federated.FedAvg(client_models, server_model)
        
        test_acc = []
        for data in test_loader:
            
            x, y = data
            x, y = x.to(DEVICE), y.to(DEVICE)
            acc = misc.accuracy(server_model(x), y)
            test_acc.append(acc)
        
        avg_testacc = sum(test_acc)/len(test_acc)    
        writer.add_scalar('FederatedSNN/globaltrainacc', avg_testacc, epoch)  
        print(f"GLobal epoch: test acc {avg_testacc}\n")
        
        
if __name__ == "__main__":
    
    
    writer = SummaryWriter(comment=f"LR_{config['lr']}_EPOCH_{config['epoch']}_FederatedSNN_{10}")
    FederatedSNN_experiment(10) 