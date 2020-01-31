"""
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)
"""
import torch
from ..models.abstracts import TextClassificationAbstract
from ..representation import get, is_transformer

class ZAGCNN(TextClassificationAbstract):
    def __init__(self, classes,   adjacency, label_embedding=None, representation="roberta", max_len=600, dropout = 0.5, **kwargs):
        super(ZAGCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.use_dropout = dropout
        self.filters = 300
        self.kernel_sizes = [3,5,10]
        self.hidden_dim = label_embedding.shape[-1]
        self.adjacency = torch.nn.Parameter(torch.from_numpy(adjacency))
        self.adjacency.requires_grad = False

        assert not is_transformer(representation), "This model wont work with a transformer Model at the moment"
        self.embedding, self.tokenizer = get(representation)
        self.embedding_dim = self.embedding.weight.shape[-1]

        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])
        self.pool = torch.nn.MaxPool1d(3, stride=2)
        from mlmc.layers import LabelAttention
        self.label_attention = LabelAttention(self.n_classes, self.filters, hidden_dim=self.hidden_dim,
                                              label_repr=label_embedding, freeze=True)
        self.dropout= torch.nn.Dropout(0.4)
        import torch_geometric as torchg
        self.gcn1 = torchg.nn.GCNConv(in_channels=label_embedding.shape[-1], out_channels=self.hidden_dim)
        self.gcn2 = torchg.nn.GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        torch.nn.MaxPool1d(3, stride=3)
        self.projection = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim+label_embedding.shape[-1])
        self.build()

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        c = torch.cat([self.pool(torch.nn.functional.relu(conv(embedded.permute(0,2,1)))) for conv in self.convs], dim=-1).permute(0,2,1)
        c, _ = self.label_attention(c)
        c = self.dropout(c)
        labelgcn = self.gcn1(self.label_attention.label_repr, torch.stack(torch.where(self.adjacency==1),dim=0))
        labelgcn = self.dropout(labelgcn)
        labelgcn = self.gcn2(labelgcn, torch.stack(torch.where(self.adjacency==1),dim=0))
        labelvectors = torch.cat([self.label_attention.label_repr, labelgcn], dim=-1)
        return (torch.relu(self.projection(c))*labelvectors).sum(-1)
