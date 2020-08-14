"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from ..abstracts_graph import TextClassificationAbstractGraph
import networkx as nx


class LSANOriginalTransformerNoClassesLAE(TextClassificationAbstractGraph):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, method="glove", scale="nothing", norm=False, representation="roberta", dropout=0.3,  d_a=200, max_len=400, n_layers=4, **kwargs):
        super(LSANOriginalTransformerNoClassesLAE, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.representation=representation
        self.method = method
        self.scale = scale
        self.norm = norm
        self.dropout=dropout
        self.n_layers=n_layers


        # Original
        self.n_classes = len(classes)
        self.representation = representation
        self._init_input_representations()

        self.create_labels(classes)


        self.label_representation_size = self.label_embedding_dim

        self.projection = torch.nn.Linear(self.embeddings_dim, self.label_representation_size * 4).to(self.device)

        self.linear_first = torch.nn.Linear(self.label_representation_size * 4, d_a).to(self.device)
        self.linear_second = torch.nn.Linear(self.label_representation_size, d_a)

        self.weight1 = torch.nn.Linear(self.label_representation_size * 2, 1).to(self.device)
        self.weight2 = torch.nn.Linear(self.label_representation_size * 2, 1).to(self.device)

        self.output_layer = torch.nn.Linear(self.label_representation_size * 2, 1)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)

        self.label_linear_first = torch.nn.Linear(in_features=self.label_representation_size,
                                                  out_features=self.label_embedding_dim).to(self.device)
        self.label_linear_second = torch.nn.Linear(in_features=self.label_embedding_dim, out_features=1).to(self.device)
        self.label_input_projection = torch.nn.Linear(in_features=self.label_embedding_dim,
                                                      out_features=self.label_representation_size).to(self.device)
        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.build()

    def init_hidden(self, size):
        return (torch.randn(2, size, self.label_embedding_dim).to(self.device),
                torch.randn(2, size, self.label_embedding_dim).to(self.device))

    def forward(self, x, return_weights=False):
        if self.finetune:
            if self.n_layers == 1:
                embeddings = self.embedding(x)[0]
            else:
                embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)

            # self.label_embedding = self.embedding(self.tokenizer(self.classes.keys()).to(self.device))[1]
        else:
            with torch.no_grad():
                if self.n_layers == 1:
                    embeddings = self.embedding(x)[0]
                else:
                    embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)

        label = self.dropout_layer(self.label_embedding)
        outputs = self.projection( self.dropout_layer(embeddings))

        # Label Embedding Creation
        labelselfatt = torch.softmax(self.label_linear_second(
            torch.tanh(self.label_linear_first(label))

        ),-2)
        label = (label*labelselfatt).sum(1)


        # step2 get self-attention
        # Label Specific Self Attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = torch.matmul(selfatt, self.linear_second(label).t())
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs[:,:,:2*self.label_representation_size])

        h1 = outputs[:, :, :self.label_representation_size]
        h2 = outputs[:, :, self.label_representation_size:2*self.label_representation_size]

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
        doc = self.embedding_dropout(doc)
         # = torch.sum(doc, -1)

        pred = self.output_layer(doc / self.label_embedding_dim).squeeze()
        if return_weights:
            return (pred, (weight1,weight2))
        return pred

    def create_label_dict(self):
        # assert method in ("repeat","generate","embed", "glove", "graph"), 'method has to be one of ("repeat","generate","embed")'
        from ...representation import Embedder
        with torch.no_grad():
            e = Embedder("glove300")
            transformed = e.embed(self.classes, pad=10)
        self.label_embedding_dim = transformed[0].shape[-1]

        # if scale == "mean":
        #     print("subtracting mean")
        #     mean = torch.cat(transformed).mean(0, keepdim=True)
        #     transformed = [l-mean for l in transformed]
        # if scale == "normalize":
        #     print("normalizing")
        #     transformed = [l / l.norm(p=2, dim=-1, keepdim=True) for l in transformed]

        if self.scale == "mean":
            print("subtracting mean")
            transformed = transformed - transformed.mean(0, keepdim=True)
        if self.scale == "normalize":
            print("normalizing")
            transformed = transformed / transformed.norm(p=2, dim=-1, keepdim=True)
            transformed[torch.isnan(transformed)] = 0

        return {w: e for w, e in zip(self.classes, transformed)}
    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = len(classes)

        if not hasattr(self, "label_dict"):
            self.label_dict = self.create_label_dict()

        from copy import deepcopy
        self.label_subgraph = deepcopy(self.kb)
        assert all([x in self.kb for x in
                    self.classes.keys()]), "If a non dynamic graph is used, all classes have to be present in the graph."
        self.label_subgraph.remove_nodes_from([x for x in self.label_subgraph if x not in self.classes.keys()])
        tmp_adj = torch.from_numpy(nx.adjacency_matrix(self.label_subgraph).toarray()).float()
        assert all([sorted([list(self.label_subgraph.nodes).index(x[1]) for x in
                            list(self.label_subgraph.edges(list(self.label_subgraph.nodes)[i]))]) == sorted(
            torch.where(tmp_adj[i] != 0)[0].tolist()) for i in range(
            len(self.label_subgraph))]), "A conversion error between graph adjacency and embedding has happened"

        tmp_adj[tmp_adj != 0] = 1
        self.adj = torch.stack(torch.where(tmp_adj == 1), dim=0).to(self.device)
        self.adjacency = tmp_adj



        self.label_embedding = torch.stack([self.label_dict[cls] for cls in classes.keys()])
        self.label_embedding = self.label_embedding.to(self.device)

    def shuffle(self):
        new_ind = torch.randperm(self.n_classes)
        new_classes = dict(zip(list(self.classes.keys()),new_ind.tolist()))
        self.create_labels(new_classes)