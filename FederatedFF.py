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

from networks.Model import FFNet, BPNet
from dataloaders.dataset import MNIST_loaders, debug_loaders
from utils import misc


torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda:2')
config = {
    'lr': 0.001,
    'epoch': 100,
    'globalepoch': 100
}


def FedAvg(Userlist, Global_model):
    """
    :param w: the list of user
    :return: the userlist after aggregated and the global mode
    """


    l_user = len(Userlist)    # the number of user

    client_weights = [1/l_user for i in range(l_user)]
    with torch.no_grad():
        for key in Global_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                Global_model.state_dict()[key].data.copy_(
                    Userlist[0].state_dict()[key])
            else:
                temp = torch.zeros_like(
                    Global_model.state_dict()[key], dtype=torch.float32)

                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * \
                        Userlist[client_idx].state_dict()[
                        key]

                Global_model.state_dict()[
                    key].data.copy_(temp)

                for client_idx in range(len(client_weights)):
                    Userlist[client_idx].state_dict()[key].data.copy_(
                        Global_model.state_dict()[key])
    return Userlist, Global_model



def FederatedFF_experiment(num_of_clients):

    num_of_clients = num_of_clients
    
    transform = tortra.Compose([
            tortra.ToTensor(),
            tortra.Normalize((0.1307,), (0.3081,)),
            tortra.Lambda(lambda x: torch.flatten(x))
            ])
    train_loader, test_loader = MNIST_loaders(batch_size=50000, transform=transform, num_subsets=num_of_clients)
    client_step = [iter(_) for _ in train_loader]
    
    server_model = FFNet([784, 500, 500]).to(DEVICE) 
    client_models = [copy.copy(server_model) for _ in range(num_of_clients)]

    
    
    FF_start_time = time.time()
    train_acc = []
    epoch = 0
    
    outertqdm = tqdm(range(config['globalepoch']), desc=f"Global Epoch", position=0)
    
    for epoch in outertqdm:
        print(f"Global epoch {epoch+1}")
        inertqdm = tqdm(train_loader, desc=f"Local client", position=1, leave=False)
        for i, iterator in enumerate(inertqdm):
            for data in iterator:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)
                x_pos = misc.overlay_y_on_x(x, y)
                rnd = torch.randperm(x.size(0))
                x_neg = misc.overlay_y_on_x(x, y[rnd])
                client_models[i].ftrain(x_pos, x_neg)
                acc = client_models[i].predict(x).eq(y).float().mean().item()
                train_acc.append(acc)
        
        client_models, server_model = FedAvg(client_models, server_model)
        
        test_acc = []
        for data in test_loader:
            
            x, y = data
            x, y = x.to(DEVICE), y.to(DEVICE)
            acc = server_model.predict(x).eq(y).float().mean().item()
            test_acc.append(acc)
        
        avg_testacc = sum(test_acc)/len(test_acc)    
        writer.add_scalar('FederatedFF/globaltrainacc', avg_testacc, epoch)  
        print(f"GLobal epoch: test acc {avg_testacc}\n")
           
            
            
            
     

    # print(f'Epoch {i} train acc of FF:', sum(train_acc)/len(train_acc))
    # FF_end_time = time.time()
    # journey = FF_end_time - FF_start_time
    # writer.add_scalar('FFAccuracy/train', sum(train_acc)/len(train_acc))
    # writer.add_scalar('Time/FFtime', journey)

    # acc_list = []
    # for x_te, y_te in test_loader:

    #     x_te, y_te = x_te.to(DEVICE), y_te.to(DEVICE)
    #     acc = net.predict(x_te).eq(y_te).float().mean().item()
    #     acc_list.append(acc)
    #     break

    # print('test acc of FF:', sum(acc_list)/len(acc_list))

    # writer.add_scalar('FFAccuracy/test', sum(acc_list)/len(acc_list))
    

if __name__ == "__main__":
    
    for client in range(20,110,10):
        writer = SummaryWriter(comment=f"LR_{config['lr']}_EPOCH_{config['epoch']}_FederatedFF_{client}")
        FederatedFF_experiment(client) 