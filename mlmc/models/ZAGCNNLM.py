"""
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)
"""
import torch
from mlmc.models.abstracts.abstracts_graph import TextClassificationAbstractGraph
from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
import re
import networkx as nx


class ZAGCNNLM(TextClassificationAbstractGraph, TextClassificationAbstractZeroShot):
    def __init__(self, classes, representation="roberta", max_len=200, dropout = 0.5, norm=False, n_layers=1, **kwargs):
        super(ZAGCNNLM, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.use_dropout = dropout
        self.filters = 300
        self.hidden_dim=512
        self.kernel_sizes = [3,4,5,6]
        self.dropout = dropout
        self.n_layers=n_layers
        self.norm = norm
        self.representation = representation
        self.graph = kwargs["graph"]

        self._init_input_representations()
        self.label_dict = self.create_label_dict()
        self.create_labels(classes)

        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.embeddings_dim, self.filters, k) for k in self.kernel_sizes])
        self.pool = torch.nn.MaxPool1d(3, stride=2)
        self.document_projection = torch.nn.Linear(self.filters, self.label_embeddings_dim)


        self.dropout_layer= torch.nn.Dropout(self.dropout)
        import torch_geometric as torchg
        self.gcn1 = torchg.nn.GCNConv(in_channels=self.label_embeddings.shape[-1], out_channels=self.hidden_dim)
        self.gcn2 = torchg.nn.GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.projection = torch.nn.Linear(in_features=self.filters, out_features=self.hidden_dim+self.label_embeddings.shape[-1])
        self.build()

    def forward(self, x):
        embeddings = self.embed_input(x)
        embedded = self.dropout_layer(embeddings)
        c = torch.cat([self.pool(torch.nn.functional.relu(conv(embedded.permute(0,2,1)))) for conv in self.convs], dim=-1).permute(0,2,1)
        d2 = torch.tanh(self.document_projection(c))
        if self.norm: d2 = d2/d2.norm(p=2,dim=-1,keepdim=True)
        a = torch.softmax(torch.matmul(d2, self.label_embeddings.t()), -1)
        label_wise_representation = torch.matmul(a.permute(0, 2, 1), c)

        label_wise_representation = self.dropout_layer(label_wise_representation)

        labelgcn = self.gcn1(self.label_embeddings, self.adj)
        labelgcn = self.dropout_layer(labelgcn)
        labelgcn = self.gcn2(labelgcn, self.adj)
        labelvectors = torch.cat([self.label_embeddings, labelgcn], dim=-1)
        return (torch.relu(self.projection(label_wise_representation)) * labelvectors).sum(-1)

    def create_label_dict(self):
        # assert method in ("repeat","generate","embed", "glove", "graph"), 'method has to be one of ("repeat","generate","embed")'
        from ..representation import get_word_embedding_mean
        with torch.no_grad():
            l = get_word_embedding_mean(
                [" ".join(re.split("[/ _-]", x.lower())) for x in self.classes.keys()],
                "glove300")
        self.label_embeddings_dim = l.shape[-1]
        return  {w:e for w,e in zip(self.classes, l)}

    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = len(classes)

        ind = [list(self.kb.nodes).index(k) for k in classes.keys()]
        tmp_adj = torch.from_numpy(nx.adjacency_matrix(self.kb).toarray()).float()

        tmp_adj = torch.stack([tmp_adj[i] for i in ind])
        tmp_adj = torch.stack([tmp_adj[:, i] for i in ind], 1)
        tmp_adj = tmp_adj + torch.eye(tmp_adj.shape[0])
        self.adj = torch.stack(torch.where(tmp_adj.t() == 1), dim=0).to(self.device)


        try:
            self.label_embeddings = torch.stack([self.label_dict[cls] for cls in classes.keys()])
        except:
            self.label_dict = self.create_label_dict()
            self.label_embeddings = torch.stack([self.label_dict[cls] for cls in classes.keys()])
        self.label_embeddings = self.label_embeddings.to(self.device)
