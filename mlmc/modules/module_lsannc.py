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
