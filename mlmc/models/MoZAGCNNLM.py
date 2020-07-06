"""
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)
"""
import torch
from .abstracts_mo import TextClassificationAbstractMultiOutput
from ..representation import get, is_transformer
import re
import networkx as nx

class MoZAGCNNLM(TextClassificationAbstractMultiOutput):
    def __init__(self, classes, graph, descriptions=None,representation="roberta", max_len=200, dropout = 0.5, norm=False, n_layers=4, **kwargs):
        super(MoZAGCNNLM, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = [len(x) for x in classes]
        self.max_len = max_len
        self.use_dropout = dropout
        self.filters = 100
        self.hidden_dim=512
        self.kernel_sizes = [3,4,5,6]
        self.dropout = dropout
        self.n_layers=n_layers
        self.norm = norm
        self.representation = representation
        self._init_input_representations()
        self.graph = graph
        self.descriptions = descriptions

        tmp_adj = torch.from_numpy(nx.adjacency_matrix(self.graph).toarray()).float()
        tmp_adj = tmp_adj + torch.eye(tmp_adj.shape[0])

        self.adj = torch.stack(torch.where(tmp_adj.t() == 1), dim=0).to(self.device)

        self.create_labels(classes, descriptions)


        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(self.embeddings_dim, self.filters, k) for k in self.kernel_sizes])
        self.pool = torch.nn.MaxPool1d(3, stride=2)
        self.document_projection = torch.nn.Linear(self.embeddings_dim, self.embeddings_dim).to(self.device)


        self.dropout_layer= torch.nn.Dropout(self.dropout)
        import torch_geometric as torchg
        self.gcn1 = torchg.nn.GCNConv(in_channels=self.embeddings_dim, out_channels=self.hidden_dim).to(self.device)
        self.gcn2 = torchg.nn.GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)

        self.projection = torch.nn.ModuleList([torch.nn.Linear(in_features=self.embeddings_dim, out_features=self.hidden_dim+self.embeddings_dim) for x in self.n_classes]).to(self.device)
        self.build()


    def forward(self, x):
        if self.finetune:
            if self.n_layers == 1:
                embeddings = self.embedding(x)[0]
            else:
                embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)

            label_embeddings = [self.embedding(x)[0].mean(-2) for x in self.label_embeddings]
        else:
            with torch.no_grad():
                if self.n_layers == 1:
                    embeddings = self.embedding(x)[0]
                else:
                    embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)
                label_embeddings = [self.embedding(x)[0].mean(-2)  for x in self.label_embeddings]

        embedded = self.dropout_layer(embeddings)
        # c = torch.cat([self.pool(torch.nn.functional.relu(conv(embedded.permute(0,2,1)))) for conv in self.convs], dim=-1).permute(0,2,1)
        d2 = torch.tanh(self.document_projection(embeddings))
        # if self.norm: d2 = d2/d2.norm(p=2,dim=-1,keepdim=True)
        a = torch.softmax(torch.matmul(d2, torch.cat(tuple(label_embeddings),0).t()), -1)
        label_wise_representation = torch.matmul(a.permute(0, 2, 1), embedded)

        label_wise_representation = self.dropout_layer(label_wise_representation)

        labelgcn = self.gcn1(torch.cat(tuple(label_embeddings),0), self.adj)
        labelgcn = self.dropout_layer(labelgcn)
        labelgcn = self.gcn2(labelgcn, self.adj)
        labelvectors = torch.cat([torch.cat(tuple(label_embeddings),0), labelgcn], dim=-1)
        labelvectors =[labelvectors[sum(self.n_classes[:i]):sum(self.n_classes[:(i+1)])] for i in range(len(self.n_classes))]
        label_wise_representation = [label_wise_representation[:,sum(self.n_classes[:i]):sum(self.n_classes[:(i+1)])] for i in range(len(self.n_classes))]


        r = [(torch.relu(p(self.dropout_layer(lr))) * lv[None,]).sum(-1) for p, lv, lr in zip(self.projection, labelvectors,label_wise_representation)]
        return r

    def create_labels(self, classes, descriptions=None):
        # assert method in ("repeat","generate","embed", "glove", "graph"), 'method has to be one of ("repeat","generate","embed")'
        self.classes = classes
        self.descriptions = descriptions

        if self.descriptions is None:
            self.label_embeddings = [self.tokenizer(x.keys(),5).to(self.device) for x in self.classes]
        else:
            self.label_embeddings = [self.tokenizer([self.descriptions[k.split("_")[0]] for k in x.keys()],5).to(self.device) for x in self.classes]

        self.label_embeddings_dim = self.label_embeddings[0].shape[-1]

    def freeze_embedding(self):
        for param in self.embedding.parameters(): param.requires_grad=False
        self.finetune = False
        self.rebuild()

    def rebuild(self, optimizer_params=None):
        if optimizer_params is not None:
            self.optimizer_params=optimizer_params
        assert isinstance(self.loss, list), "loss has not been build yet."
        if self.class_weights is not None:
            self.loss = [self.loss.__class__(torch.FloatTensor(w).to(self.device)) for loss, w in zip(self.loss,self.class_weights)]
        else:
            self.loss = [loss.__class__() for loss in self.loss]
        if isinstance(self.optimizer, type) and self.optimizer is not None:
            self.optimizer = self.optimizer.__class__(filter(lambda p: p.requires_grad, self.parameters()),
                                            **self.optimizer_params)
        self.to(self.device)
