"""
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)
"""
import torch
from ..models.abstracts import TextClassificationAbstract
from ..representation import get, is_transformer
import re

class ZAGCNN(TextClassificationAbstract):
    def __init__(self, classes,   adjacency, representation="roberta", max_len=300, dropout = 0.5, **kwargs):
        super(ZAGCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.use_dropout = dropout
        self.filters = 300
        self.hidden_dim = 300
        self.kernel_sizes = [3,10]
        self.adjacency=adjacency
        self.dropout = dropout
        self.adjacency_parameter = torch.nn.Parameter(torch.from_numpy(adjacency).float())
        self.adjacency_parameter.requires_grad = False

        assert not is_transformer(representation), "This model does not support language Models. See ZAGCNNLM."
        self.representation = representation
        self._init_input_representations()
        self.embedding.requires_grad=False
        self.label_embeddings = torch.nn.Parameter(
            self.embedding(self.tokenizer(
                [" ".join(re.split("[/ _-]", x.lower())) for x in self.classes.keys()]

            )[:, :4]).mean(-2)
        )
        self.label_embeddings.requires_grad = False


        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.embedding_dim, self.filters, k) for k in self.kernel_sizes])
        self.pool = torch.nn.MaxPool1d(3, stride=2)
        self.document_projection = torch.nn.Linear(self.filters, self.embedding_dim).to(self.device)



        self.dropout_layer= torch.nn.Dropout(self.dropout)
        import torch_geometric as torchg
        self.gcn1 = torchg.nn.GCNConv(in_channels=self.embedding_dim, out_channels=self.hidden_dim)
        self.gcn2 = torchg.nn.GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)

        self.projection = torch.nn.Linear(in_features=self.embedding_dim, out_features=self.hidden_dim+self.label_embeddings.shape[-1])
        self.build()

    def forward(self, x):
        with torch.no_grad():
            embedded = self.embedding(x)
        embedded = self.dropout_layer(embedded)
        c = torch.cat([self.pool(torch.nn.functional.relu(conv(embedded.permute(0,2,1)))) for conv in self.convs], dim=-1).permute(0,2,1)
        d2 = torch.tanh(self.document_projection(c))
        a = torch.softmax(torch.matmul(d2, self.label_embeddings.t()),-1)

        label_wise_representation = torch.matmul(a.permute(0,2,1), c)

        label_wise_representation = self.dropout_layer(label_wise_representation)

        labelgcn = self.gcn1(self.label_embeddings, torch.stack(torch.where(self.adjacency_parameter == 1), dim=0))
        labelgcn = self.dropout_layer(labelgcn)
        labelgcn = self.gcn2(labelgcn, torch.stack(torch.where(self.adjacency_parameter == 1), dim=0))
        labelvectors = torch.cat([self.label_embeddings, labelgcn], dim=-1)
        return (torch.relu(self.projection(label_wise_representation))*labelvectors).sum(-1)

    def create_labels(self, classes):
        self.classes = classes
        del self.label_embeddings
        self.label_embeddings = torch.nn.Parameter(
            self.embedding(self.tokenizer([" ".join(re.split("[/ _-]", x.lower()) ) for x in self.classes.keys()])[:, :4].to(self.device)).mean(-2)
        )
        self.label_embeddings.requires_grad = False

    def set_adjacency(self, adj):
        del self.adjacency_parameter
        self.adjacency_parameter = torch.nn.Parameter(torch.from_numpy(adj).float().to(self.device))
        self.adjacency_parameter.requires_grad = False