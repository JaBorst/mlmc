import torch

def cos(x,y):
    return torch.matmul((x/x.norm(p=2,dim=-1,keepdim=True)) , (y/y.norm(p=2,dim=-1,keepdim=True)).t())

