import torch
from . import NC_LabelSelfAttention, NC_LabelSpecificSelfAttention, DynamicWeightedFusion, SplitWrapper


class LSANNCModule(torch.nn.Module):
    """
    This Module corresponds to the model from the Paper "Label-specific Document Representations for document classification",
    but in class number independend formulation.
    It is formulated as a module to return the LSAN representations as described in the paper (Not the classification result).
    """

    def __init__(self, in_dim, node_dim, hidden_features=512, noise=0.00, **kwargs):
        """
        Class constructor.

        :param in_dim: Size of each input sample
        :param node_dim: Size of each label embedding
        :param hidden_features: Hidden state size
        :param noise: Noise for regularization
        """
        super(LSANNCModule, self).__init__()
        self.noise = noise
        self.in_dim = in_dim
        self.node_dim = node_dim
        self.hidden_features = hidden_features

        assert in_dim % 2 == 0, "in_dim has to even"
        self.half_dim = int(self.in_dim / 2)

        # self.label_projection = torch.nn.Linear(self.node_dim, self.half_dim)
        self.x_projection = torch.nn.Linear(self.in_dim, self.node_dim * 2)

        self.lsa = SplitWrapper(self.node_dim, NC_LabelSelfAttention())
        self.lssa = NC_LabelSpecificSelfAttention(self.node_dim * 2, self.node_dim, self.hidden_features)
        self.dwf = DynamicWeightedFusion(self.node_dim * 2, 2, noise=self.noise)

    def forward(self, x, nodes, mask=None, return_weights=False):
        """
        Forward pass function for transforming input tensor into output tensor. Creates the label-specific document
        representation along all labels.

        :param x: Input tensor
        :param nodes: Tensor containing label embeddings
        :param return_weights: If true, returns the learnable weights of the module as well
        :return: Output tensor
        """
        x = self.x_projection(x)
        i = self.lssa(x, nodes)
        l = self.lsa(x, nodes)
        d, w = self.dwf([l, i])
        if return_weights:
            return d, w
        else:
            return d


class LSANVocabModule(torch.nn.Module):
    """
    This is an LSAN inspired Module which compares against an input against a set of representations
    but returns a tensor with the same dimensions as the input.
    It's a scale conserving operation.
    """

    def __init__(self, in_dim, node_dim, hidden_features=512, noise=0.01):
        """
        Class constructor.

        :param in_dim: Size of each input sample
        :param node_dim: Size of each label embedding
        :param hidden_features: Hidden state size
        :param noise: Noise for regularization
        """
        super(LSANVocabModule, self).__init__()
        self.hidden_features = hidden_features
        self.node_dim = node_dim
        self.in_dim = in_dim
        self.noise = noise
        self.dwf = DynamicWeightedFusion(in_dim, 2, noise=self.noise)

        self.linear_first = torch.nn.Linear(in_dim, self.hidden_features)
        self.linear_second = torch.nn.Linear(node_dim, self.hidden_features)
        self.projection = torch.nn.Linear(self.node_dim, self.in_dim)
        self.projection2 = torch.nn.Linear(self.in_dim, self.node_dim)

    def forward(self, x, nodes, return_weights=False, mask=None):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :param nodes: Tensor containing label embeddings
        :param return_weights: If true, returns the learnable weights of the module as well
        :return: Output tensor
        """
        nodes_p = self.projection(nodes)
        selfatt = torch.tanh(self.linear_first(x))
        selfatt = torch.matmul(selfatt, self.linear_second(nodes).t())
        selfatt = torch.softmax(selfatt, dim=-1)
        self_att = torch.matmul(selfatt, nodes_p)
        m1 = torch.softmax(torch.matmul(self.projection2(x), nodes.t()), -1)
        label_att = torch.matmul(m1, nodes_p)
        fused, w = self.dwf([self_att, label_att])
        if return_weights:
            return fused, w
        return fused

    def regularize(self):
        """Regularization function"""
        return self.linear_first.weight.data.norm(p=2) + self.linear_second.weight.data.norm(p=2) + \
               self.projection.weight.data.norm(p=2) + self.projection2.weight.data.norm(p=2)


class LSANGraphModule(torch.nn.Module):
    """
       This is an LSAN inspired Module which compares against an input against a
       a graph to produce transformed representations of the input.
        It returns a tensor with the same dimensions as the input.
        It's a scale conserving operation.
    """

    def __init__(self, in_dim, node_dim, hidden_features=512, n_layers=3, gtype="GraphConv", noise=0.01, **kwargs):
        """
        Class constructor.

        :param in_dim: Size of each input sample
        :param node_dim: Size of each label embedding
        :param hidden_features: Hidden state size
        :param n_layers: Number of convolutional layers
        :param gtype: Determines the convolutional layer used. GCNConv if graph_type="gcn", else GatedGraphConv
        :param noise: Noise for regularization
        """
        super(LSANGraphModule, self).__init__()
        self.hidden_features = hidden_features
        self.node_dim = node_dim
        self.in_dim = in_dim
        self.n_layers = n_layers
        self.noise = noise
        self.dwf = DynamicWeightedFusion(in_dim, 2, noise=self.noise)

        self.linear_first = torch.nn.Linear(in_dim, self.hidden_features)
        self.linear_second = torch.nn.Linear(self.in_dim, self.hidden_features)
        self.gtype = gtype
        import torch_geometric as torchg
        if self.gtype == "GraphConv":
            self.gcn = torch.nn.ModuleList(
                [torchg.nn.GraphConv(in_channels=self.node_dim, out_channels=self.in_dim, aggr="mean") if i == 0 else
                 torchg.nn.GraphConv(in_channels=self.in_dim, out_channels=self.in_dim, aggr="mean")
                 for i in range(n_layers)])
        elif self.gtype == "GATConv":
            self.gcn = torch.nn.ModuleList(
                [torchg.nn.GATConv(in_channels=self.node_dim, out_channels=self.in_dim) if i == 0 else
                 torchg.nn.GATConv(in_channels=self.in_dim, out_channels=self.in_dim)
                 for i in range(n_layers)])
        self.projection = torch.nn.Linear(self.node_dim, self.in_dim)

    def forward(self, x, nodes, graph, weights=None, return_weights=False, mask=None):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :param nodes: Tensor containing label embeddings
        :param graph: Tensor containing graph adjacency matrix
        :param weights: Tensor containing edge weights
        :param return_weights: If true, returns the learnable weights of the module as well
        :return: Output tensor
        """
        nodes_p = nodes
        if self.gtype == "GraphConv":
            for l in self.gcn:
                nodes_p = l(x=nodes_p, edge_index=graph, edge_weight=weights)
        elif self.gtype == "GATConv":
            for l in self.gcn:
                nodes_p = l(x=nodes_p, edge_index=graph)

        selfatt = torch.tanh(self.linear_first(x))
        selfatt = torch.matmul(selfatt, self.linear_second(nodes_p).t())
        selfatt = torch.softmax(selfatt, dim=-1)
        self_att = torch.matmul(selfatt, nodes_p)

        m1 = torch.softmax(torch.matmul(x, nodes_p.t()), -1)
        label_att = torch.matmul(m1, nodes_p)
        self.entropy = (-(selfatt * torch.log(selfatt)).sum(-1)).mean() + (-(m1 * torch.log(m1)).sum(-1)).mean()
        fused, w = self.dwf([self_att, label_att])
        if return_weights:
            return fused, w
        return fused

    def regularize(self):
        """Regularization function"""
        return self.linear_first.weight.data.norm(p=2) + self.linear_second.weight.data.norm(p=2)
