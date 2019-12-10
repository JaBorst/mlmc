import torch

class Prob(torch.nn.Module):
    def __init__(self, n_labels):
        super(Prob,self).__init__()
        self.corr = torch.nn.Parameter(torch.randn(n_labels, n_labels))
    def forward(self, x):
        x = (torch.unsqueeze(x, -1) * self.corr).sum(-2)
        return x

    