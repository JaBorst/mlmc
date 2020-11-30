"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
from mlmc.models.abstracts.abstracts_graph import TextClassificationAbstractGraph
import re
import networkx as nx
from mlmc.modules.layer_nc_attention_comparison import *


class LSAN3BFG(TextClassificationAbstractGraph):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes,share_weighting=False, hidden_representations=400, weight_norm ="norm", branch_noise = 0.001, representation="roberta", dropout=0.3, propagation_layers=3, d_a=200, max_len=400, n_layers=4, **kwargs):
        super(LSAN3BFG, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.representation=representation
        self.n_layers=n_layers
        self.propagation_layers = propagation_layers
        self.dropout = dropout
        self.branch_noise = branch_noise
        self.weight_norm = weight_norm
        self.share_weighting = share_weighting
        # Original
        self.n_classes = len(classes)
        self.representation = representation
        self._init_input_representations()

        self.label_dict = self.create_label_dict()
        self.create_labels(classes)
        self.log_bw = False
        self.bw=[]
        self.d_a = d_a
        self.hidden_representations=hidden_representations


        self.projection_input = torch.nn.Linear(self.embeddings_dim, self.hidden_representations * 2)
        self.projection_labels = torch.nn.Linear(self.label_embedding_dim, self.hidden_representations)
        self.lsatt = NC_LabelSpecificSelfAttention(in_features=self.hidden_representations * 2, in_features2=self.hidden_representations, hidden_features=self.d_a)
        self.latt = SplitWrapper(self.hidden_representations, NC_LabelSelfAttention(hidden_features=self.hidden_representations))
        self.dynamic_fusion = DynamicWeightedFusion(in_features=self.hidden_representations * 2, n_inputs=3, share_weights=share_weighting)
        self.gsa =SplitWrapper(self.hidden_representations, GraphSpecificSelfAttention(hidden_features=self.hidden_representations, n_layers=3))
        self.output_layer = torch.nn.Linear(self.hidden_representations * 2, 1)

        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.build()
    def forward(self, x, return_weights=False):
        outputs = self.projection_input(self.embed_input(x))
        label_embed = self.projection_labels(self.label_embedding)

        self_att = self.lsatt(outputs, label_embed)[:,:self.n_classes]
        label_att = self.latt(outputs, label_embed)[:,:self.n_classes]
        graph_att = self.gsa(outputs, label_embed, self.dgl_graph)[:,:self.n_classes]

        doc, weights = self.dynamic_fusion([self_att, label_att, graph_att])

        doc = self.dropout_layer(doc)
        pred = self.output_layer(doc / self.label_embedding_dim).squeeze()
        if self.log_bw:
            self.bw.append(weights.cpu())
        return pred

    def log_branch_weights(self, s=True):
        self.log_bw=s
    def reset_branch_weights(self):
        self.bw=[]
    def get_branch_weights(self):
        return torch.cat(self.bw).cpu()

    def create_label_dict(self):
        from ...representation import get_word_embedding_mean
        with torch.no_grad():
            l = get_word_embedding_mean(
                    [" ".join(re.split("[/ _-]", x.lower())) for x in self.kb.nodes()],
                    "glove300")
        self.label_embedding_dim = l.shape[-1]
        return {w: e for w, e in zip(self.kb.nodes, l)}

    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = len(classes)
        assert all([x in self.kb for x in
                    self.classes.keys()]), "If a non dynamic graph is used, all classes have to be present in the graph."
        self.label_subgraph = nx.OrderedDiGraph()
        self.label_subgraph.add_nodes_from(self.classes.keys())
        self.label_subgraph.add_nodes_from(self.kb.nodes)
        self.label_subgraph.add_edges_from(self.kb.edges)
        nx.set_node_attributes(self.label_subgraph, dict(self.kb.nodes(True)))
        nx.set_edge_attributes(self.label_subgraph, {(x[0],x[1]): x[2] for x in self.kb.edges(data=True)})

        tmp_adj = torch.from_numpy(nx.adjacency_matrix(self.label_subgraph).toarray()).float()
        assert all([sorted([list(self.label_subgraph.nodes).index(x[1]) for x in
                            list(self.label_subgraph.edges(list(self.label_subgraph.nodes)[i]))]) == sorted(
            torch.where(tmp_adj[i] != 0)[0].tolist()) for i in range(
            len(self.label_subgraph))]), "A conversion error between graph adjacency and embedding has happened"
        assert list(self.label_subgraph.nodes)[:len(self.classes)] == list(self.classes),"Sorting went wrong"
        tmp_adj[tmp_adj != 0] = 1
        self.adj = torch.stack(torch.where(tmp_adj == 1), dim=0).to(self.device)
        self.dgl_graph = dgl.graph((list(self.adj[0].cpu()), list(self.adj[1].cpu())))
        self.dgl_graph = dgl.add_self_loop(self.dgl_graph).to(self.device)

        try:
            self.label_embedding = torch.stack([self.label_dict[cls] for cls in self.label_subgraph.nodes])
        except:
            self.create_label_dict(method=self.method, scale=self.scale)
            self.label_embedding = torch.stack([self.label_dict[cls] for cls in self.label_subgraph.nodes])
        self.label_embedding = self.label_embedding.to(self.device)

    def shuffle(self):
        new_ind = torch.randperm(self.n_classes)
        new_classes = dict(zip(list(self.classes.keys()),new_ind.tolist()))
        self.create_labels(new_classes)