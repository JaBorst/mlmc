import torch
import torch_geometric as torchg

from mlmc.modules.layer_nc_attention_comparison import NC_LabelSpecificSelfAttention


class ZAGCNNModule(torch.nn.Module):
    def __init__(self, in_features, in_features2, hidden_features=512, graph_type="gcn", propagation_layers=3,
                 dropout=0.5):
        """
        Class constructor and initialization of every hyperparameter.

        :param in_features: Number of inputs for first linear layer
        :param in_features2: Number of inputs for second linear layer
        :param hidden_features: Hidden state dimension
        :param graph_type: Determines the convolutional layer used. GCNConv if graph_type="gcn", else GatedGraphConv
        :param propagation_layers: Number of GCNConv layers. Only used when graph_type="gcn"
        :param dropout: Dropout rate
        """
        super(ZAGCNNModule, self).__init__()

        self.graph_type = graph_type
        self.propagation_layers = propagation_layers
        self.in_features2 = in_features2
        self.in_features = in_features
        self.hidden_features = hidden_features

        if self.graph_type == "gcn":
            self.gcn1 = torch.nn.ModuleList(
                [torchg.nn.GCNConv(in_channels=self.in_features2, out_channels=self.in_features2, node_dim=0)
                 for i in range(self.propagation_layers)])
        else:
            self.ggc = torchg.nn.GatedGraphConv(self.in_features2, num_layers=self.in_features2, node_dim=0)

        self.lssa = NC_LabelSpecificSelfAttention(in_features=in_features, in_features2=in_features2,
                                                  hidden_features=hidden_features)
        self.dropout_layer = torch.nn.Dropout(0.5)
        self.leaky_relu = torch.nn.LeakyReLU()
        self.projection = torch.nn.Linear(in_features=self.in_features2 * 2, out_features=self.in_features)

    def forward(self, x, nodes, adjacency):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :param nodes: Nodes of the graph
        :param adjacency: Adjacency matrix of the graph
        :return: Output tensor
        """
        # nodes wise label attention
        e = self.lssa(x, nodes)

        graph_nodes = nodes
        if self.graph_type == "gcn":
            for m in self.gcn1:
                graph_nodes = m(graph_nodes, adjacency)
                graph_nodes = self.dropout_layer(self.leaky_relu(graph_nodes))
        else:
            graph_nodes = torch.stack([self.ggc(x=x, edge_index=adjacency) for x in graph_nodes])

        labelvectors = torch.cat([nodes, graph_nodes], dim=-1)
        return (torch.relu(e) * self.projection(labelvectors))
