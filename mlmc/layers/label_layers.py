"""Layers that are concerned with scoring labels"""

import torch


class LabelAttention(torch.nn.Module):
    """Reimplementation of label attention as described in paper: Label-specific Document representations.
    Might be deprecated.
    """
    def __init__(self, n_classes, input_dim, hidden_dim, label_repr=None, freeze=True):
        super(LabelAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        if self.hidden_dim is not None:
            if self.hidden_dim != self.input_dim:
                self.projection = torch.nn.Linear(self.input_dim, self.hidden_dim)

        if label_repr is None:
            self.label_repr = torch.nn.Parameter(torch.Tensor(n_classes, self.hidden_dim))
            torch.nn.init.kaiming_normal_(self.label_repr)
        else:
            assert label_repr.shape[-1] == hidden_dim,"label embedding dimension must equal hidden_dim"
            self.label_repr = torch.nn.Parameter(label_repr)
            self.label_repr.requires_grad=freeze

    def forward(self, x):
        if self.hidden_dim is not None:
            if self.hidden_dim != self.input_dim:
                x = self.projection(x)
        A =torch.softmax(torch.matmul(x, self.label_repr.permute(1,0)),-1)
        output = torch.matmul(A.permute(0,2,1), x)
        return output, A

class LabelEmbeddingAttention(torch.nn.Module):
    def __init__(self, n_classes, input_dim, hidden_dim, label_embedding):
        super(LabelEmbeddingAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        if self.hidden_dim is not None:
            if self.hidden_dim != self.input_dim:
                self.projection = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.label_repr = label_embedding

    def forward(self, x):
        A =  torch.softmax(torch.matmul(x, self.label_repr.permute(1,0)),-1)
        output = torch.matmul(A.permute(0,2,1), x)
        return output, A


class LabelEmbeddingScoring(torch.nn.Module):
    """Layer that keeps a representation (static Embedding) and compares the input to all vectors. The metric
        should be freely choosable
    """
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

class LabelSpecificSelfAttention(torch.nn.Module):
    def __init__(self, n_classes, input_dim, hidden_dim):
        super(LabelSpecificSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        # self.to_hidden = torch.nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.to_label = torch.nn.Parameter(torch.Tensor(self.hidden_dim, self.n_classes))

        self.to_hidden = torch.nn.Linear(self.input_dim, self.hidden_dim)
        # self.to_label = torch.nn.Linear(self.hidden_dim, self.n_classes)

        # torch.nn.init.kaiming_normal_(self.to_hidden)
        torch.nn.init.kaiming_normal_(self.to_label)

    def forward(self,x):
        att = torch.softmax(torch.matmul(
            torch.tanh(self.to_hidden(x)),self.to_label),
            -1)
        # att = torch.softmax(self.to_label(torch.tanh(self.to_hidden(x))), -1)
        return torch.matmul(att.permute(0,2,1),x), att

class AdaptiveCombination(torch.nn.Module):
    def __init__(self, input_dim, n_classes ):
        super(AdaptiveCombination, self).__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        self.alpha_weights = torch.nn.Linear(input_dim,1)
        self.beta_weights = torch.nn.Linear(input_dim,1)
        # torch.nn.init.kaiming_normal_(self.alpha_weights)
        # torch.nn.init.kaiming_normal_(self.beta_weights)

    def forward(self, x):
        # alpha = torch.sigmoid(torch.matmul(x[0], self.alpha_weights))
        # beta = torch.sigmoid(torch.matmul(x[1], self.beta_weights))

        alpha = torch.sigmoid(self.alpha_weights(x[0]))
        beta = torch.sigmoid(self.beta_weights(x[1]))

        #constrain the sum to one
        alpha = alpha / (alpha + beta)
        beta = beta / (alpha + beta)
        output = alpha*x[0] + beta*x[1]
        return output
