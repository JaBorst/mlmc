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


class LSANOriginalTransformerNoClassesHierarchical3BranchXGraph(TextClassificationAbstractGraph):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, method="glove", scale="mean", norm=False, representation="roberta", dropout=0.3, propagation_layers=3, d_a=200, max_len=400, n_layers=4, **kwargs):
        super(LSANOriginalTransformerNoClassesHierarchical3BranchXGraph, self).__init__(**kwargs)
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


        self.projection = torch.nn.Linear(self.embeddings_dim, self.label_embedding_dim * 4).to(self.device)

        self.linear_first = torch.nn.Linear(self.label_embedding_dim * 4, d_a).to(self.device)
        self.linear_second = torch.nn.Linear(self.label_embedding_dim , d_a)

        self.weight1 = torch.nn.Linear(self.label_embedding_dim * 2, 1).to(self.device)
        self.weight2 = torch.nn.Linear(self.label_embedding_dim * 2, 1).to(self.device)
        self.weight3 = torch.nn.Linear(self.label_embedding_dim * 2, 1).to(self.device)

        self.output_layer = torch.nn.Linear(self.label_embedding_dim * 2, 1)

        # self.connection = torch.nn.Linear(self.label_embedding_dim * 2, 1)
        import torch_geometric as  torchg
        # self.label_embedding_gcn = torch.nn.ModuleList([torchg.nn.GCNConv(in_channels=self.label_embedding_dim, out_channels=self.label_embedding_dim,
        #                                              node_dim=0).to(self.device) for _ in range(self.propagation_layers)])
        self.ggc = torchg.nn.GatedGraphConv(out_channels=self.label_embedding_dim, num_layers=propagation_layers, node_dim=0).to(self.device)


        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.build()

    def forward(self, x):
        # if self.training: self.shuffle()

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

        embeddings = self.dropout_layer(embeddings)
        # step1 get LSTM outputs
        # hidden_state = self.init_hidden(x.shape[0])
        outputs = self.projection(embeddings)#, hidden_state)
        # step2 get self-attention


        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = torch.matmul(selfatt, self.linear_second(self.label_embedding[:self.n_classes]).t())
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs[:,:,:2*self.label_embedding_dim])
        # step3 get label-attention

        h1 = outputs[:, :, :self.label_embedding_dim]
        h2 = outputs[:, :, self.label_embedding_dim:2*self.label_embedding_dim]
        h3 = outputs[:, :, 2*self.label_embedding_dim:3*self.label_embedding_dim]
        h4 = outputs[:, :, 3*self.label_embedding_dim:]


        label1 = self.label_embedding[:self.n_classes]
        m1 = torch.bmm(label1.expand(x.shape[0], * label1.shape), h1.transpose(1, 2))
        m2 = torch.bmm(label1.expand(x.shape[0], * label1.shape), h2.transpose(1, 2))
        label_att = torch.relu(torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2))

        #
        #
        # for l in self.label_embedding_gcn:
        #     label = l(label, self.adj)
        #     label = self.dropout_layer(label)
        label2 = self.label_embedding
        label2 = self.ggc(label2, self.adj)
        graph1 = torch.bmm(label2.expand(x.shape[0], *label2.shape), h3.transpose(1, 2))
        graph2 = torch.bmm(label2.expand(x.shape[0], *label2.shape), h4.transpose(1, 2))
        graph_att = torch.relu(torch.cat((torch.bmm(graph1, h1), torch.bmm(graph2, h2)), 2))
        graph_att = graph_att[:,:self.n_classes]

        weight1 = torch.sigmoid(self.weight1(label_att))
        weight2 = torch.sigmoid(self.weight2(self_att))
        weight3 = torch.sigmoid(self.weight3(graph_att))

        norm =  (weight1 + weight2 + weight3)
        weight1 = weight1 /norm
        weight3 = weight3 / norm
        weight2 = 1 - weight1 - weight3

        doc = weight1 * label_att + weight2 * self_att + weight3*graph_att
        # there two method, for simple, just add
        # also can use linear to do it
        doc = self.dropout_layer(doc)
         # = torch.sum(doc, -1)

        pred = self.output_layer(doc / self.label_embedding_dim).squeeze()
        return pred

    def create_label_dict(self, scale="nothing"):

        from ...representation import get_word_embedding_mean
        from ...graph import subgraphs, get
        self.label_subgraph_base = subgraphs(self.classes, self.kb, model="glove300", topk=self.topk,
                                        depth=self.depth, device=self.device)

        self.label_subgraph_base = self.label_subgraph_base.to_undirected(reciprocal=False)
        with torch.no_grad():
            l = get_word_embedding_mean(
                [" ".join(re.split("[/ _-]", x.lower())) for x in self.label_subgraph_base.nodes],
                "glove300")
        if scale == "mean":
            print("subtracting mean")
            l = l - l.mean(0, keepdim=True)
        if scale == "normalize":
            print("normalizing")
            l = l / l.norm(p=2, dim=-1, keepdim=True)
        self.label_embedding_dim = l.shape[-1]
        return {w: e for w, e in zip(self.label_subgraph_base.nodes, l)}

    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = len(classes)
        from copy import deepcopy
        if not hasattr(self, "label_dict"):
            self.label_dict = self.create_label_dict(scale=self.scale)

        self.label_subgraph = deepcopy(self.label_subgraph_base)
        self.label_subgraph.remove_nodes_from([x for x,v in dict(self.label_subgraph.nodes(True)).items() if x not in self.classes.keys() and v.get("type","None")=="label"])

        tmp_adj = torch.from_numpy(nx.adjacency_matrix(self.label_subgraph).toarray()).float()
        assert all([sorted([list(self.label_subgraph.nodes).index(x[1]) for x in
                            list(self.label_subgraph.edges(list(self.label_subgraph.nodes)[i]))]) == sorted(
            torch.where(tmp_adj[i] != 0)[0].tolist()) for i in range(
            len(self.label_subgraph))]), "A conversion error between graph adjacency and embedding has happened"

        tmp_adj[tmp_adj != 0] = 1
        self.adj = torch.stack(torch.where(tmp_adj == 1), dim=0).to(self.device)
        self.adjacency = tmp_adj


        self.label_embedding = torch.stack([self.label_dict[cls] for cls in self.label_subgraph.nodes])
        self.label_embedding = self.label_embedding.to(self.device)



    def shuffle(self):
        new_ind = torch.randperm(self.n_classes)
        new_classes = dict(zip(list(self.classes.keys()),new_ind.tolist()))
        self.create_labels(new_classes)