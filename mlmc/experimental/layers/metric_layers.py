import torch

class Metric(torch.nn.Module):
    """Experimental feature A simple weighted vector sum.
        Idea:
            x[i, j] every activation i of the input corresponds to a distribution over the output class j
            The distributions are weighted and summed.
    """
    def __init__(self, dim):
        super(Bilinear, self).__init__()
        self.dim = dim
        self.metric_tensor = torch.nn.Parameter(
            torch.triu(torch.zeros((dim, dim)))
        )
        self.metric_tensor.requires_grad=True
        torch.nn.init.eye_(self.metric_tensor)


    def forward(self, x, y):
        assert len(y.shape)==2, "y has to a"
        metric = torch.matmul(self.metric_tensor.triu(), self.metric_tensor.triu().t())
        # differences = (x.unsqueeze(-1) - y.t().unsqueeze(0)).permute(0,2,1)
        # p = torch.matmul(differences, metric)
        # p = (differences*p).sum(-1)
        return torch.matmul(torch.matmul(x,metric.t()),y.t())

    def regularize(self):
        lower_tri = torch.triu(self.metric_tensor).t()
        dev = torch.relu((lower_tri.sum(-1) - 2 * lower_tri.diag()))
        return dev.sum() + torch.relu(-lower_tri.diag()).sum()



class Bilinear(torch.nn.Module):
    """Experimental feature A simple weighted vector sum.
        Idea:
            x[i, j] every activation i of the input corresponds to a distribution over the output class j
            The distributions are weighted and summed.
    """
    def __init__(self, dim):
        super(Bilinear, self).__init__()
        self.dim = dim
        self.metric_tensor = torch.nn.Parameter(
            torch.triu(torch.zeros((dim, dim)))
        )
        self.metric_tensor.requires_grad=True
        torch.nn.init.eye_(self.metric_tensor)


    def forward(self, x, y):
        assert len(y.shape)==2, "y has to have 2 dimensions"
        metric = torch.matmul(self.metric_tensor.triu(), self.metric_tensor.triu().t())
        return torch.matmul(torch.matmul(x,metric.t()),y.t())

    def regularize(self):
        lower_tri = torch.triu(self.metric_tensor).t()
        dev = torch.relu((lower_tri.sum(-1) - 2 * lower_tri.diag()))
        return dev.sum() + torch.relu(-lower_tri.diag()).sum()


class Metric(torch.nn.Module):
    """Experimental feature A simple weighted vector sum.
        Idea:
            x[i, j] every activation i of the input corresponds to a distribution over the output class j
            The distributions are weighted and summed.
    """
    def __init__(self, dim):
        super(Metric, self).__init__()
        self.dim = dim
        self.metric_tensor = torch.nn.Parameter(
            torch.triu(torch.zeros((dim, dim)))
        )
        self.metric_tensor.requires_grad=True
        torch.nn.init.eye_(self.metric_tensor)


    def forward(self, x, y):
        assert len(y.shape)==2, "y has to a"
        metric = torch.matmul(self.metric_tensor.triu(), self.metric_tensor.triu().t())
        n_unsqueezes = len(x.shape)-2
        m=y.t()
        for i in range(n_unsqueezes):
            m = m.unsqueeze(0)
        differences = (x.unsqueeze(-1) - m).permute(0,2,1)
        p = torch.matmul(differences, metric)
        p = (differences*p).sum(-1)
        return p

    def regularize(self):
        lower_tri = torch.triu(self.metric_tensor).t()
        dev = torch.relu((lower_tri.sum(-1) - 2 * lower_tri.diag()))
        return dev.sum() + torch.relu(-lower_tri.diag()).sum()

