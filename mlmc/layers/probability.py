import torch

class Prob(torch.nn.Module):
    """Experimental feature A simple weighted vector sum.
        Idea:
            x[i, j] every activation i of the input corresponds to a distribution over the output class j
            The distributions are weighted and summed.
    """
    def __init__(self, n_labels):
        super(Prob,         self).__init__()
        self.corr = torch.nn.Parameter(torch.Tensor(n_labels, n_labels))
        torch.nn.init.eye_(self.corr)

    def forward(self, x):
        x = (torch.unsqueeze(x, -1) * self.corr).sum(-2)
        return x
