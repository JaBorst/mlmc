# import torch
#
# class Prob(torch.nn.Module):
#     def __init__(self, n_labels):
#         super(Prob,self).__init__()
#         self.corr = torch.nn.Parameter(torch.randn(n_labels, n_labels))
#     def forward(self, x):
#         x = (torch.unsqueeze(x, -1) * self.corr).sum(-2)
#         return x
#
#

import numpy as np
import torch

class LabelEmbeddingScoring(torch.nn.Module):
    def __init__(self, n_classes, input_dim, label_repr, similarity="cosine", label_freeze=True):
        super(LabelEmbeddingScoring, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        assert similarity in ["cosine","euclidean"], "Distance metric %s not implemented." % (similarity, )
        self.similarity=similarity

        self.label_repr = torch.nn.Parameter(torch.from_numpy(label_repr).float())
        self.label_repr.requires_grad=not label_freeze
        self.projection = torch.nn.Linear(self.input_dim, self.label_repr.shape[-1])

    def forward(self, x):
        x = self.projection(x)

        if self.similarity=="cosine":
            output = torch.matmul(
                x/torch.norm(x,p=2,dim=-1).unsqueeze(-1),
                (self.label_repr/torch.norm(self.label_repr, p=2,dim=-1).unsqueeze(-1)).transpose(0,1)
            )
        if self.similarity=="euclidean":
            output = torch.sigmoid(
                torch.norm((x.unsqueeze(2) - self.label_repr.unsqueeze(0).unsqueeze(1)),p=2,dim=-1)
            )
        return output



l = LabelEmbeddingScoring(52, 400,np.random.rand(52,50))
l(torch.Tensor(np.random.rand(16,140, 400)))
