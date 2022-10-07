import torch

class VerticalDropout(torch.nn.Module):
    def __init__(self, dropout=0.1):
        super(VerticalDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                mask = torch.zeros_like(x[:,:,0,None]).uniform_() > self.dropout
            x = x + mask* torch.rand_like(x)
        return x

