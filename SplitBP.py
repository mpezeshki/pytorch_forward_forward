
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from networks.Model import  BPNet
from dataloaders.dataset import MNIST_loaders
from utils import misc

DEVICE = torch.device('cuda')
config = {
    'lr': 0.001,
    'epoch': 50,

}
writer = SummaryWriter(
    comment=f"LR_{config['lr']}_EPOCH_{config['epoch']}_rewriteFF")


# def BP_experiment():
#     BPtrain_loader, BPtest_loader = MNIST_loaders(512, 512)

#     BP_net = BPNet([784, 500, 10]).to(DEVICE)
#     BP_loss = nn.CrossEntropyLoss(reduction='none')
#     BP_optim = torch.optim.Adam(BP_net.parameters(), lr=config['lr'])
#     bestacc = 0.0
#     minloss = 1.0
#     journeylist = []
#     for epoch in range(config['epoch']):
#         BP_start_time = time.time()
#         BP_metric = misc.Accumulator(3)
#         for x, y in tqdm(BPtrain_loader):
#             if isinstance(BP_net, torch.nn.Module):
#                 BP_net.train()
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             y_hat = BP_net(x)
#             loss = BP_loss(y_hat, y)
#             BP_optim.zero_grad()
#             loss.mean().backward()
#             BP_optim.step()

#             BP_metric.add(float(loss.sum()),
#                           misc.accuracy(y_hat, y), y.numel())
#         BP_end_time = time.time()
#         journeylist.append(BP_end_time-BP_start_time)

#         avg_loss, avg_acc = BP_metric[0] / \
#             BP_metric[2], BP_metric[1] / BP_metric[2]
#         writer.add_scalar('BPLoss/train', avg_loss, epoch)
#         writer.add_scalar('BPAccuracy/train', avg_acc, epoch)
#         print(f"Epoch {epoch}: loss {avg_loss}, acc {avg_acc}\n")
#         if avg_loss <= minloss:
#             minloss = avg_loss
#         if avg_acc >= bestacc:
#             bestacc = avg_acc

#         BP_net.eval()
#         testmetric = misc.Accumulator(2)
#         with torch.no_grad():
#             for X, y in BPtest_loader:
#                 X = X.to(DEVICE)
#                 y = y.to(DEVICE)
#                 testmetric.add(misc.accuracy(BP_net(X), y), y.numel())

#         testacc = testmetric[0] / testmetric[1]
#         writer.add_scalar('BPAccuracy/test', testacc, epoch)

#     writer.add_scalar('Time/BPtime', sum(journeylist))

#     print(f"BP trainacc:{bestacc} \n BP trainloss:{minloss}")
#     print(f"BP testacc:{testacc}\n")

# BPtrain_loader, BPtest_loader = MNIST_loaders(512, 512)





DEVICE = torch.device('cuda')
config = {
    'lr': 0.001,
    'epoch': 50,
}

writer = SummaryWriter(comment=f"LR_{config['lr']}_EPOCH_{config['epoch']}_splitNN")

BPtrain_loader, BPtest_loader = MNIST_loaders(512)

BP_net = BPNet_split([784, 500, 10]).to(DEVICE)
BP_loss = nn.CrossEntropyLoss(reduction='none')
BP_optim = torch.optim.Adam(BP_net.parameters(), lr=config['lr'])
journeylist = []
bestacc = 0.0
minloss = 1.0

for epoch in range(config['epoch']):
    BP_start_time = time.time()
    BP_metric = misc.Accumulator(3)

    for x, y in tqdm(BPtrain_loader[0], leave=False):
        if isinstance(BP_net, torch.nn.Module):
            BP_net.train()
        x, y = x.to(DEVICE), y.to(DEVICE)

        # Shallow training pass
        shallow_output = BP_net.shallow_model(x)
        deep_output = BP_net.deep_model(shallow_output)
        loss = BP_loss(deep_output, y)
        BP_optim.zero_grad()
        loss.mean().backward()
        BP_optim.step()

        BP_metric.add(float(loss.sum()), misc.accuracy(deep_output, y), y.numel())

    BP_end_time = time.time()
    journeylist.append(BP_end_time - BP_start_time)

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
            X, y = X.to(DEVICE), y.to(DEVICE)
            shallow_output = BP_net.shallow_model(X)
            deep_output = BP_net.deep_model(shallow_output)
            testmetric.add(misc.accuracy(BP_net(X), y), y.numel())

    testacc = testmetric[0] / testmetric[1]
    writer.add_scalar('BPAccuracy/test', testacc, epoch)

writer.add_scalar('Time/BPtime', sum(journeylist))

print(f"BP trainacc:{bestacc} \n BP trainloss:{minloss}")
print(f"BP testacc:{testacc}\n")

