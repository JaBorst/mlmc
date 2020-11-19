import torch
from dgl.nn.pytorch import GraphConv
import dgl

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
    def __init__(self, hidden_features):
        super(NC_LabelSelfAttention, self).__init__()
        self.hidden_features = hidden_features
        self.act = torch.nn.LeakyReLU(negative_slope = 0.01, inplace = False)

    def forward(self, x, label):
        m1 = torch.softmax(torch.bmm(label.expand(x.shape[0], *label.shape), x.transpose(1, 2)),-1)
        label_att = torch.bmm(m1, x)
        return torch.relu(label_att)


class GraphSpecificSelfAttention(torch.nn.Module):
    def __init__(self, hidden_features, n_layers=3, dropout=0.2):
        super(GraphSpecificSelfAttention, self).__init__()
        self.hidden_features = hidden_features
        self.n_layers = n_layers
        self.dropout = dropout
        self.gcn = torch.nn.ModuleList(
            [GraphConv(in_feats=self.hidden_features, out_feats=self.hidden_features)
             for _ in range(self.n_layers)]
        )
        self.act = torch.nn.LeakyReLU(negative_slope = 0.01, inplace = False)
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, nodes, graph: dgl.DGLGraph):
        for l in self.gcn:
            nodes = l(graph, nodes)
            nodes = self.dropout_layer(nodes)
        graph_att = torch.bmm(nodes.expand(x.shape[0], *nodes.shape), x.transpose(1, 2))
        graph_att = torch.bmm(graph_att, x)
        return torch.relu(graph_att)

class SplitWrapper(torch.nn.Module):
    def __init__(self, size, layer: torch.nn.Module):
        super(SplitWrapper, self).__init__()
        self.layer = layer
        self.size = size
    def forward(self, x, *args, **kwargs):
        h1, h2 = x[:,:,:self.size], x[:,:,self.size:]
        return torch.cat([self.layer(h1,*args, **kwargs), self.layer(h2,*args, **kwargs)], -1)

class DynamicWeightedFusion(torch.nn.Module):
    def __init__(self, in_features, n_inputs, noise=0.01, dropout=0.2, norm="softmax", share_weights=False):
        super(DynamicWeightedFusion, self).__init__()
        self.in_features = in_features
        self.n_inputs = n_inputs
        self.noise = noise
        self.dropout = dropout
        self.norm = norm
        self.share_weights = share_weights
        assert norm in ("softmax","sigmoid","norm")
        if self.share_weights:
            self.weights_projection = torch.nn.Linear(in_features,1)
        else:
            self.weights_projection = torch.nn.ModuleList(
                [torch.nn.Linear(in_features,1) for _ in range(self.n_inputs)])

    def forward(self, x):
        if self.share_weights:
            weights = torch.cat([self.weights_projection(i) for i in x],-1)
        else:
            weights = torch.cat([p(i) for p,i in zip(self.weights_projection, x)],-1)
        if self.training:
            noise = self.noise * torch.rand(weights.size()).to(weights.device)
            weights = weights + noise
        if self.norm == "softmax":
            weights = torch.softmax(weights, -1)
        elif self.norm == "norm":
            weights = torch.sigmoid(weights)
            weights = weights[:,:,:1] / weights.sum(-1, keepdim=True)
        elif self.norm == "sigmoid":
            weights = torch.sigmoid(weights)

        return (torch.stack(x,-2) * weights.unsqueeze(-1)).sum(-2), weights

