"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from ..abstracts import TextClassificationAbstract
from ..abstracts_zeroshot import TextClassificationAbstractZeroShot
from ..abstracts_graph import TextClassificationAbstractGraph
import re
import networkx as nx


class LSANOriginalTransformerNoClassesHierarchicalXGraph(TextClassificationAbstractGraph):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, method="glove", scale="mean", norm=False, representation="roberta", dropout=0.3, propagation_layers=3, d_a=200, max_len=400, n_layers=4, **kwargs):
        super(LSANOriginalTransformerNoClassesHierarchicalXGraph, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.representation=representation
        self.method = method
        self.scale = scale
        self.norm = norm
        self.n_layers=n_layers
        self.propagation_layers = propagation_layers
        self.dropout = dropout

        # Original
        self.n_classes = len(classes)
        self.representation = representation
        self._init_input_representations()

        self.create_labels(classes)

        tmp_adj = torch.from_numpy(nx.adjacency_matrix(self.kb).toarray()).float()
        assert all([sorted([list(self.kb.nodes).index(x[1]) for x in
                            list(self.kb.edges(list(self.kb.nodes)[i]))]) == sorted(
            torch.where(tmp_adj[i] != 0)[0].tolist()) for i in range(
            len(self.kb))]), "A conversion error between graph adjacency and embedding has happened"

        tmp_adj[tmp_adj != 0] = 1
        self.adj = torch.stack(torch.where(tmp_adj == 1), dim=0).to(self.device)
        self.adjacency = tmp_adj


        self.projection = torch.nn.Linear(self.embeddings_dim, self.label_embeddings_dim * 2)

        self.linear_first = torch.nn.Linear(self.label_embeddings_dim * 2, d_a)
        self.linear_second = torch.nn.Linear(self.label_embeddings_dim , d_a)

        self.weight1 = torch.nn.Linear(self.label_embeddings_dim * 2, 1)
        self.weight2 = torch.nn.Linear(self.label_embeddings_dim * 2, 1)

        self.output_layer = torch.nn.Linear(self.label_embeddings_dim * 2, 1)

        # self.connection = torch.nn.Linear(self.label_embedding_dim * 2, 1)
        import torch_geometric as  torchg
        self.label_embedding_gcn = torch.nn.ModuleList([torchg.nn.GATConv(in_channels=self.label_embeddings_dim, out_channels=self.label_embeddings_dim,
                                                     node_dim=0).to(self.device) for _ in range(self.propagation_layers)])

        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.kernel_sizes = [3, 4, 5, 6,7]
        self.filters = 100
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.label_embeddings_dim, self.filters, k) for k in self.kernel_sizes]).to(self.device)
        self.label_projection = torch.nn.Linear(in_features=len(self.kernel_sizes)*self.filters, out_features=self.label_embeddings_dim)
        self.build()

    def forward(self, x):
        if self.finetune:
            if self.n_layers == 1:
                embeddings = self.embedding(x)[0]
                label = self.embedding(self.label_embeddings)[0]

            else:
                embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)
                label = self.embedding(self.label_embeddings)[0]
        else:
            with torch.no_grad():
                if self.n_layers == 1:
                    embeddings = self.embedding(x)[0]
                    label = self.embedding(self.label_embeddings)[0]
                else:
                    embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)
                    label = self.embedding(self.label_embeddings)[0]

        # label = label.mean(-2)
        c = [torch.nn.functional.relu(conv(label.permute(0,2,1)).permute(0, 2, 1).max(1)[0]) for conv in self.convs]
        label = torch.cat(c, 1)
        label = self.label_projection(self.dropout_layer(label))
        for l in self.label_embedding_gcn:
            label = l(label, self.adj)
            label = self.dropout_layer(label)

        if self.n_classes < self.no_nodes: label = label[:self.n_classes]

        # step1 get LSTM outputs
        # hidden_state = self.init_hidden(x.shape[0])
        outputs = self.projection(embeddings)#, hidden_state)
        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = torch.matmul(selfatt, self.linear_second(label).t())
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        # step3 get label-attention

        h1 = outputs[:, :, :self.label_embeddings_dim]
        h2 = outputs[:, :, self.label_embeddings_dim:]


        m1 = torch.bmm(label.expand(x.shape[0], *label.shape), h1.transpose(1, 2))
        m2 = torch.bmm(label.expand(x.shape[0], *label.shape), h2.transpose(1, 2))
        label_att = torch.relu(torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2))

        weight1 = torch.sigmoid(self.weight1(label_att))
        weight2 = torch.sigmoid(self.weight2(self_att))


        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1

        doc = weight1 * label_att + weight2 * self_att
        # there two method, for simple, just add
        # also can use linear to do it
        doc = self.dropout_layer(doc)
         # = torch.sum(doc, -1)

        pred = self.output_layer(doc / self.label_embeddings_dim).squeeze()
        return pred
