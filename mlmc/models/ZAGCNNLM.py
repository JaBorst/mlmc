"""
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)
"""
import torch
from ..models.abstracts import TextClassificationAbstract
from ..representation import get, is_transformer

class ZAGCNNLM(TextClassificationAbstract):
    def __init__(self, classes,   adjacency, representation="roberta", max_len=200, dropout = 0.5, **kwargs):
        super(ZAGCNNLM, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.use_dropout = dropout
        self.filters = 300
        self.hidden_dim=512
        self.kernel_sizes = [3,10]
        self.dropout = dropout
        self.adjacency = adjacency
        self.adjacency_param = torch.nn.Parameter(torch.from_numpy(adjacency).float())
        self.adjacency_param.requires_grad = False
        self.n_layers=3
        self.representation = representation
        self._init_input_representations()

        self.label_embeddings = torch.nn.Parameter(self.embedding(self.tokenizer(list(self.classes.keys())))[1])
        self.label_embeddings.requires_grad = False
        self.label_embeddings_dim = self.label_embeddings.shape[-1]

        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])
        self.pool = torch.nn.MaxPool1d(3, stride=2)
        self.document_projection = torch.nn.Linear(self.filters, self.label_embeddings_dim)


        self.dropout_layer= torch.nn.Dropout(self.dropout)
        import torch_geometric as torchg
        self.gcn1 = torchg.nn.GCNConv(in_channels=self.label_embeddings.shape[-1], out_channels=self.hidden_dim)
        self.gcn2 = torchg.nn.GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.projection = torch.nn.Linear(in_features=self.filters, out_features=self.hidden_dim+self.label_embeddings.shape[-1])
        self.build()

    def forward(self, x):
        with torch.no_grad():
            embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
        embedded = self.dropout_layer(embeddings)
        c = torch.cat([self.pool(torch.nn.functional.relu(conv(embedded.permute(0,2,1)))) for conv in self.convs], dim=-1).permute(0,2,1)
        d2 = torch.tanh(self.document_projection(c))
        a = torch.softmax(torch.matmul(d2, self.label_embeddings.t()), -1)
        label_wise_representation = torch.matmul(a.permute(0, 2, 1), c)

        label_wise_representation = self.dropout_layer(label_wise_representation)

        labelgcn = self.gcn1(self.label_embeddings, torch.stack(torch.where(self.adjacency_param == 1), dim=0))
        labelgcn = self.dropout_layer(labelgcn)
        labelgcn = self.gcn2(labelgcn, torch.stack(torch.where(self.adjacency_param == 1), dim=0))
        labelvectors = torch.cat([self.label_embeddings, labelgcn], dim=-1)
        return (torch.relu(self.projection(label_wise_representation)) * labelvectors).sum(-1)
