import torch

class AttentionWeightedAggregation(torch.nn.Module):
    """
    Aggregate a tensor weighted by weights obtained from self attention
    https://arxiv.org/pdf/1703.03130.pdf
    """
    def __init__(self, in_features, d_a):
        super(AttentionWeightedAggregation, self).__init__()
        self.in_features=in_features
        self.d_a = d_a

        self.att_projection = torch.nn.Linear(in_features=self.in_features, out_features=self.d_a, bias=False)
        self.att_weights = torch.nn.Parameter(torch.zeros((self.d_a,)))
        torch.nn.init.normal_(self.att_weights, mean=0.0, std=0.001)


    def forward(self, x, aggr=None, return_att=True):
        attention = torch.softmax(torch.matmul(torch.tanh(self.att_projection(x)), self.att_weights), -1)
        if aggr is None:
            output = (attention[:, :, None]*x).sum(-2)
        else:
            output = (attention[:, :, None]*aggr).sum(-2)
        if return_att:
            return output, attention
        else:
            return output

