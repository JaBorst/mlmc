"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
from mlmc.models.abstracts.abstracts_graph import TextClassificationAbstractGraph
import re
import networkx as nx
from mlmc.modules.layer_nc_attention_comparison import *
import dgl

class LSAN3BAG2(TextClassificationAbstractGraph):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, method="glove", scale="mean",hidden_representations=400,share_weighting=False, weight_norm ="norm", branch_noise = 0.001, representation="roberta", dropout=0.3, propagation_layers=3, d_a=200, max_len=400, n_layers=4, **kwargs):
        super(LSAN3BAG2, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.representation=representation
        self.method = method
        self.scale = scale
        self.n_layers=n_layers
        self.propagation_layers = propagation_layers
        self.dropout = dropout
        self.branch_noise = branch_noise
        self.weight_norm = weight_norm
        self.d_a=d_a
        self.hidden_representations = hidden_representations
        self.share_weighting = share_weighting


        # Original
        self.n_classes = len(classes)
        self.representation = representation
        self._init_input_representations()

        self.label_dict = self.create_label_dict()
        self.create_labels(classes)
        self.log_bw = False
        self.bw=[]



        tmp_adj = torch.from_numpy(nx.adjacency_matrix(self.kb).toarray()).float()
        assert all([sorted([list(self.kb.nodes).index(x[1]) for x in
                            list(self.kb.edges(list(self.kb.nodes)[i]))]) == sorted(
            torch.where(tmp_adj[i] != 0)[0].tolist()) for i in range(
            len(self.kb))]), "A conversion error between graph adjacency and embedding has happened"

        tmp_adj[tmp_adj != 0] = 1
        self.adj = torch.stack(torch.where(tmp_adj == 1), dim=0).to(self.device)
        self.adjacency = tmp_adj
        self.dgl_graph = dgl.graph((list(self.adj[0].cpu()), list(self.adj[1].cpu())))
        self.dgl_graph = dgl.add_self_loop(self.dgl_graph).to(self.device)

        self.projection_input = torch.nn.Linear(self.embeddings_dim, self.hidden_representations * 2).to(self.device)
        self.projection_labels = torch.nn.Linear(self.label_embedding_dim, self.hidden_representations ).to(self.device)

        self.lsatt = NC_LabelSpecificSelfAttention(in_features=self.hidden_representations * 2, in_features2=self.hidden_representations, hidden_features=self.d_a)
        self.lsatt2 = NC_LabelSpecificSelfAttention(in_features=self.hidden_representations * 2, in_features2=self.hidden_representations, hidden_features=self.d_a)

        self.lsatt2 = NC_LabelSpecificSelfAttention(in_features=self.hidden_representations, hidden_features=self.d_a).to(self.device)


        self.latt = NC_LabelSelfAttention(hidden_features=self.hidden_representations)
        self.dynamic_fusion = DynamicWeightedFusion(in_features=self.hidden_representations*2, n_inputs=4,share_weights=share_weighting)
        self.gsa = GraphSpecificSelfAttention( hidden_features=self.hidden_representations, n_layers=3)
        self.gsa2 = GraphSpecificSelfAttention( hidden_features=self.hidden_representations, n_layers=3).to(self.device)
        self.output_layer = torch.nn.Linear(self.hidden_representations * 2, 1)

        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.build()

    def forward(self, x, return_weights=False):
        outputs = self.projection_input(self.embed_input(x))
        label_embed = self.projection_labels(self.label_embedding)
        graph_embed = self.projection_labels(self.graph_embedding)



        label_graph_att = self.gsa(label_embed.repeat(1,2)[None], graph_embed, self.dgl_graph)
        label = self.latt(label_graph_att, label_embed)[0]


        self_att = self.lsatt(outputs, label[:,:self.hidden_representations])
        self_att2 = self.lsatt(outputs, label[:,self.hidden_representations:])


        label_att = self.latt(outputs, label_embed)

        graph_att = self.gsa2(outputs, graph_embed, self.dgl_graph)

        doc, weights = self.dynamic_fusion([self_att[:,:self.n_classes], self_att2[:,:self.n_classes],  label_att[:,:self.n_classes], graph_att[:,:self.n_classes]])

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
        self.label_dict={w: e for w, e in zip(self.classes, l)}


        with torch.no_grad():
            l = get_word_embedding_mean(
                [" ".join(re.split("[/ _-]", x.lower())) for x in self.kb.nodes()],
                "glove300")
        self.graph_embedding_dim = l.shape[-1]
        self.graph_embedding=l.to(self.device)

    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = len(classes)
        self.label_subgraph = self.kb

        tmp_adj = torch.from_numpy(nx.adjacency_matrix(self.label_subgraph).toarray()).float()
        tmp_adj[tmp_adj != 0] = 1
        self.adj = torch.stack(torch.where(tmp_adj == 1), dim=0).to(self.device)

        try:
            self.label_embedding = torch.stack([self.label_dict[cls] for cls in self.classes.keys()])
        except:
            self.create_label_dict()
            self.label_embedding = torch.stack([self.label_dict[cls] for cls in  self.classes.keys()])
        self.label_embedding = self.label_embedding.to(self.device)
