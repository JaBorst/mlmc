import torch


class SKGModule(torch.nn.Module):
    def __init__(self, in_features, in_features2, sequence_length, graph_type="gcn", propagation_layers=3, dropout=0.5):
        """
        Class constructor and initialization of every hyperparameter.

        :param in_features: Size of each input sample for first fully-connected layer
        :param in_features2: Size of each input sample for second fully-connected layer
        :param sequence_length: Maximum size of each input/output sample
        :param graph_type: Determines the convolutional layer used. GCNConv if graph_type="gcn", else GatedGraphConv
        :param propagation_layers: Number of GCNConv layers. Only used when graph_type="gcn"
        :param dropout: Dropout rate
        """
        super(SKGModule, self).__init__()
        self.sequence_length = sequence_length
        self.graph_type = graph_type
        self.propagation_layers = propagation_layers
        self.in_features = in_features
        self.in_features2 = in_features2
        import torch_geometric as torchg
        if graph_type == "gcn":
            self.gcn1 = torch.nn.ModuleList(
                [torchg.nn.GCNConv(in_channels=self.sequence_length, out_channels=self.sequence_length, node_dim=1)
                 for i in range(propagation_layers)])
        else:
            self.ggc = torchg.nn.GatedGraphConv(self.sequence_length, num_layers=self.sequence_length, node_dim=0)
        self.belief_projection = torch.nn.Linear(in_features=self.sequence_length, out_features=1)
        self.eye = torch.eye(self.sequence_length)[None]
        self.leaky_relu = torch.nn.LeakyReLU()
        self.projection = torch.nn.Linear(self.in_features2, self.in_features)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, x, nodes, adjacency):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :param nodes: Nodes of the graph
        :param adjacency: Adjacency matrix of the graph
        :return: Output tensor
        """
        score_matrix = torch.einsum("ijk,lmk->ijl", self.dropout_layer(x), self.projection(nodes))
        beliefs = score_matrix.transpose(1, 2)
        if self.graph_type == "gcn":
            for m in self.gcn1:
                beliefs = m(beliefs, adjacency)
                beliefs = self.dropout_layer(self.leaky_relu(beliefs))
        else:
            beliefs = torch.stack([self.ggc(x=x, edge_index=adjacency) for x in beliefs])
        beliefs = self.belief_projection(beliefs)[:, :, 0]
        return beliefs
