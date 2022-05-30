import torch
from torch import nn
from .alias_multinomial import AliasMethod
import math
import torch.nn.functional as F

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, use_softmax=True):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.use_softmax = use_softmax
        self.register_buffer('params', torch.tensor([K, T * math.sqrt(inputSize), -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        rnd = torch.randn(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        self.register_buffer('memory', F.normalize(rnd.sign(), dim=1))


    def update_memory(self, data):
        memory = 0
        for i in range(len(data)):
            memory += data[i]
        memory /= memory.norm(dim=1, keepdim=True)
        self.memory.mul_(0).add_(memory)
    
    def forward(self, l, ab, y, idx=None, epoch=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_l = self.params[2].item()
        Z_ab = self.params[3].item()

        momentum = self.params[4].item() if (epoch is None) else (0 if epoch < 0 else self.params[4].item())
        batchSize = l.size(0)
        outputSize = self.memory.size(0)
        inputSize = self.memory.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        if momentum <= 0:
            weight = (l + ab) / 2.
            inx = torch.stack([torch.arange(batchSize)] * batchSize)
            inx = torch.cat([torch.arange(batchSize).view([-1, 1]), inx[torch.eye(batchSize) == 0].view([batchSize, -1])], dim=1).to(weight.device).view([-1])
            weight = weight[inx].view([batchSize, batchSize, -1])
        else:
            weight = torch.index_select(self.memory, 0, idx.view(-1)).detach().view(batchSize, K + 1, inputSize)

        weight = weight.sign_()
        out_ab = torch.bmm(weight, ab.view(batchSize, inputSize, 1))
        # sample
        out_l = torch.bmm(weight, l.view(batchSize, inputSize, 1))
        if self.use_softmax:
            out_ab = torch.div(out_ab, T)
            out_l = torch.div(out_l, T)
            out_l = out_l.contiguous()
            out_ab = out_ab.contiguous()
        else:
            out_ab = torch.exp(torch.div(out_ab, T))
            out_l = torch.exp(torch.div(out_l, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_l < 0:
                self.params[2] = out_l.mean() * outputSize
                Z_l = self.params[2].clone().detach().item()
                print("normalization constant Z_l is set to {:.1f}".format(Z_l))
            if Z_ab < 0:
                self.params[3] = out_ab.mean() * outputSize
                Z_ab = self.params[3].clone().detach().item()
                print("normalization constant Z_ab is set to {:.1f}".format(Z_ab))
            # compute out_l, out_ab
            out_l = torch.div(out_l, Z_l).contiguous()
            out_ab = torch.div(out_ab, Z_ab).contiguous()

        # # update memory
        with torch.no_grad():
            l = (l + ab) / 2.
            l.div_(l.norm(dim=1, keepdim=True))
            l_pos = torch.index_select(self.memory, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(l, 1 - momentum))
            l_pos = l_pos.div_(l_pos.norm(dim=1, keepdim=True))
            self.memory.index_copy_(0, y, l_pos)

        return out_l, out_ab
