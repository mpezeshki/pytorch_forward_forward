import torch
import torch.nn as nn
import numpy as np

thresh, lens, decay = (0.5, 0.5, 0.2)
DEVICE = torch.device('cuda:2')


def make_paraset(nps):
    params_M = [[0.060, 0.219, -0.065, 0.003],
                [0.060, 0.219, -0.010, 0.050],
                [0.060, 0.219, -0.135, -0.051]]

    params_T = [[-0.009, 0.246, -0.058, 0.065],
                [-0.009, 0.246, 0.064, 0.091],
                [0.005, 0.158, -0.058, 0.065],
                [0.005, 0.158, 0.064, 0.091],
                [0.007, 0.205, -0.058, 0.065],
                [0.007, 0.205, 0.064, 0.091]]

    if nps == 'M1':
        paraset = [params_M[0]]
    elif nps == 'M2':
        paraset = [params_M[1]]
    elif nps == 'M3':
        paraset = [params_M[2]]
    elif nps == 'T1':
        paraset = [params_T[0]]
    elif nps == 'T2':
        paraset = [params_T[1]]
    elif nps == 'T3':
        paraset = [params_T[2]]
    elif nps == 'T4':
        paraset = [params_T[3]]
    elif nps == 'T5':
        paraset = [params_T[4]]
    elif nps == 'T6':
        paraset = [params_T[5]]

    return paraset





class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens

        return grad_input * temp.float()

act_fun = ActFun.apply

class FC_block(nn.Module):
    def __init__(self, stepsize, time_windows, lastlayer_size, input_size, output_size, if_bias = True):
        super(FC_block, self).__init__()
        self.batch_size = stepsize
        self.input_size = input_size
        self.output_size = output_size
        self.time_window = time_windows
        self.last_layer = (output_size == lastlayer_size)
        self.order = 0.001

        if self.last_layer:
            self.if_print = False
        else:
            self.if_print = False

        self.input = None
        self.v = None
        self.u = None
        self.spike = None
        self.sumspike = None
        self.ntype = 'M1'

        self.time_counter = 0

        self.fc = nn.Linear(self.input_size, self.output_size, if_bias).to(DEVICE)

        decide = False
        hsize = (self.batch_size, self.output_size)
        if decide:
            if self.ntype != '1':
                paraset = make_paraset(self.ntype)
                lens = len(paraset)
                org = lens * torch.rand((self.output_size))
                org = org.floor()
                org = org.clamp(0, lens - 1)
                num = []
                for i in range(lens):
                    num.append(0)

                for i in range(self.output_size):
                    num[int(org[i].data)] += 1

                self.a = None
                self.b = None
                self.c = None
                self.d = None

                for i in range(lens):
                    if self.a is None:
                        if num[i] != 0:
                            self.a = paraset[i][0] * torch.ones((self.batch_size, num[i]))
                            self.b = paraset[i][1] * torch.ones((self.batch_size, num[i]))
                            self.c = paraset[i][2] * torch.ones((self.batch_size, num[i]))
                            self.d = paraset[i][3] * torch.ones((self.batch_size, num[i]))
                    else:
                        if num[i] != 0:
                            self.a = torch.cat((self.a, paraset[i][0] * torch.ones((self.batch_size, num[i]))), 1)
                            self.b = torch.cat((self.b, paraset[i][1] * torch.ones((self.batch_size, num[i]))), 1)
                            self.c = torch.cat((self.c, paraset[i][2] * torch.ones((self.batch_size, num[i]))), 1)
                            self.d = torch.cat((self.d, paraset[i][3] * torch.ones((self.batch_size, num[i]))), 1)
                self.a = self.a.to(DEVICE)
                self.b = self.b.to(DEVICE)
                self.c = self.c.to(DEVICE)
                self.d = self.d.to(DEVICE)
        else:
            a, b, c, d = [0.02, 0.2, 0, 0.08]
            distime = 0
            self.a = nn.Parameter(a * (torch.ones(hsize) + distime * (torch.rand(hsize) - 0.5)))
            self.b = nn.Parameter(b * (torch.ones(hsize) + distime * (torch.rand(hsize) - 0.5)))
            self.c = nn.Parameter(c * (torch.ones(hsize) + distime * (torch.rand(hsize) - 0.5)))
            self.d = nn.Parameter(d * (torch.ones(hsize) + distime * (torch.rand(hsize) - 0.5)))
            distrue = True
            self.a.requires_grad = distrue
            self.b.requires_grad = distrue
            self.c.requires_grad = distrue
            self.d.requires_grad = distrue

    def mem_update(self, ops, x):
        ops = ops.to(DEVICE)
        self.v = self.v.to(DEVICE)
        self.spike = self.spike.to(DEVICE)
        self.u = self.u.to(DEVICE)
        self.d = self.d.to(DEVICE)
        self.a = self.a.to(DEVICE)
        self.b = self.b.to(DEVICE)
        self.c = self.c.to(DEVICE)
        self.sumspike = self.sumspike.to(DEVICE)
        
        
        if self.ntype == '1':
            I = torch.sigmoid(ops(x))
            self.v = self.v * decay * (1 - self.spike) + I
            self.spike = act_fun(self.v)
            self.sumspike = self.sumspike + self.spike
        else:
            

            self.v = self.v * (1 - self.spike) + self.spike * self.c
            self.u = self.u + self.spike * self.d
            I = torch.sigmoid(ops(x))
            v_delta = self.v * self.v - self.v - self.u + I
            u_delta = self.a * (self.b * self.v - self.u)
            self.v = self.v + v_delta
            self.u = self.u + u_delta
            self.spike = act_fun(self.v)
            self.sumspike = self.sumspike + self.spike

    def forward(self, input):
        if self.time_counter == 0:
            self.v = 0 * torch.ones((self.batch_size,self.output_size))
            self.u = 0.08 * torch.ones((self.batch_size,self.output_size))
            self.spike = torch.zeros((self.batch_size,self.output_size))
            self.sumspike = torch.zeros((self.batch_size,self.output_size))

        self.input = input
        
        self.mem_update(self.fc, self.input)

        self.time_counter += 1
        if self.time_counter == self.time_window:
            self.time_counter = 0

        return self.spike