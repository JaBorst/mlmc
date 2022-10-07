import torch

class NC_LabelSpecificSelfAttention(torch.nn.Module):
    def __init__(self, in_features, in_features2, hidden_features):
        super(NC_LabelSpecificSelfAttention, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.in_features2 = in_features2
        self.linear_first = torch.nn.Linear(self.in_features, self.hidden_features)
        self.linear_second = torch.nn.Linear(self.in_features2, self.hidden_features)

    def forward(self, x, label):
        selfatt = torch.tanh(self.linear_first(x))
        selfatt = torch.matmul(selfatt, self.linear_second(label).t())
        selfatt = torch.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, x)
        return self_att


class NC_LabelSelfAttention(torch.nn.Module):
    def __init__(self):
        super(NC_LabelSelfAttention, self).__init__()
        self.act = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x, label):
        m1 = torch.bmm(label.expand(x.shape[0], *label.shape), x.transpose(1, 2))
        label_att = torch.bmm(m1, x)
        return torch.relu(label_att)

class SplitWrapper(torch.nn.Module):
    def __init__(self, size, layer: torch.nn.Module):
        super(SplitWrapper, self).__init__()
        self.layer = layer
        self.size = size

    def forward(self, x, *args, **kwargs):
        h1, h2 = x[:, :, :self.size], x[:, :, self.size:]
        return torch.cat([self.layer(h1, *args, **kwargs), self.layer(h2, *args, **kwargs)], -1)


class DynamicWeightedFusion(torch.nn.Module):
    def __init__(self, in_features, n_inputs, noise=0.01, dropout=0.2, norm="norm", share_weights=False):
        super(DynamicWeightedFusion, self).__init__()
        self.in_features = in_features
        self.n_inputs = n_inputs
        self.noise = noise
        self.dropout = dropout
        self.norm = norm
        self.share_weights = share_weights
        assert norm in ("softmax", "sigmoid", "norm")
        if self.share_weights:
            self.weights_projection = torch.nn.Linear(in_features, 1)
        else:
            self.weights_projection = torch.nn.ModuleList(
                [torch.nn.Linear(in_features, 1) for _ in range(self.n_inputs)])

    def forward(self, x):
        if self.share_weights:
            weights = torch.cat([self.weights_projection(i) for i in x], -1)
        else:
            weights = torch.cat([p(i) for p, i in zip(self.weights_projection, x)], -1)
        if self.training:
            noise = self.noise * torch.rand(weights.size()).to(weights.device)
            weights = weights + noise
        if self.norm == "softmax":
            weights = torch.softmax(weights, -1)
        elif self.norm == "norm":
            weights = torch.sigmoid(weights)
            weights = weights / weights.sum(-1, keepdim=True)
        elif self.norm == "sigmoid":
            weights = torch.sigmoid(weights)

        return (torch.stack(x, -2) * weights.unsqueeze(-1)).sum(-2), weights
