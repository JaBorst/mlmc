import torch

def norm(x):
    return x/x.norm(p=2,dim=-1, keepdim=True)

def squash(x):
    norm = x.norm(p=2,dim=-1, keepdim=True)
    return norm**2 / (1+ norm**2) * x / norm

class Squash(torch.nn.Module):
    def forward(self, x):
        return squash(x)
