"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from ...models.abstracts import TextClassificationAbstract
from ...models.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...models.abstracts_graph import TextClassificationAbstractGraph
import re
import networkx as nx
from ...layers import ExpertCoRep

class LSAN1LM(TextClassificationAbstractGraph):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, method="glove", scale="mean",  representation="roberta", dropout=0.3, propagation_layers=3, d_a=250, max_len=200, n_layers=4, explanations=None, **kwargs):
        super(LSAN1LM, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.representation=representation
        self.method = method
        self.scale = scale
        self.n_layers=n_layers
        self.propagation_layers = propagation_layers
        self.dropout = dropout
        self.graph = "wordnet"
        self.label_text_length = 50
        self.d_a = d_a
        self.explanations = explanations
        # Original
        self.n_classes = len(classes)
        self.representation = representation
        self._init_input_representations()

        self.create_labels(classes)
        self.label_representation_size = 400

        self.linear_first = torch.nn.Linear(self.label_representation_size , d_a).to(self.device)
        self.linear_second = torch.nn.Linear(self.label_representation_size , d_a)

        self.weight1 = torch.nn.Linear(self.label_representation_size, 1).to(self.device)
        self.weight2 = torch.nn.Linear(self.label_representation_size, 1).to(self.device)

        self.output_layer = torch.nn.Linear(in_features=self.label_representation_size, out_features=self.d_a).to(self.device)
        self.output_layer2 = torch.nn.Linear(in_features=self.d_a, out_features=1)

        self.label_input_projection = torch.nn.Linear(in_features=self.label_embedding_dim, out_features=self.label_representation_size).to(self.device)
        self.embedding_input_projection = torch.nn.Linear(in_features=self.label_embedding_dim, out_features=self.label_representation_size).to(self.device)
        self.glove_input_projection = torch.nn.Linear(in_features=self.glove_dim, out_features=self.label_representation_size).to(self.device)

        self.label_transformation = torch.nn.MultiheadAttention(embed_dim=self.label_representation_size, num_heads=1).to(self.device)
        self.label_att_agg = torch.nn.Linear(in_features=150, out_features=1).to(self.device)
        self.embedding_att_agg = torch.nn.Linear(in_features=self.max_len, out_features=1).to(self.device)
        self.embedding_transformation = torch.nn.MultiheadAttention(embed_dim=self.label_representation_size, num_heads=1).to(self.device)
        self.label_projection = torch.nn.Linear(in_features=self.glove_dim+self.label_representation_size, out_features=self.label_representation_size)

        self.label_transformation2 = torch.nn.MultiheadAttention(embed_dim=self.label_representation_size, num_heads=1).to(self.device)
        self.embedding_transformation2 = torch.nn.MultiheadAttention(embed_dim=self.label_representation_size, num_heads=1).to(self.device)


        self.label_projection = torch.nn.Linear(in_features=150, out_features=self.label_representation_size).to(self.device)

        self.num_lstm_layers = 2
        self.label_lstm = torch.nn.LSTM(input_size=self.label_representation_size,dropout=0.5,
                                        hidden_size=self.label_representation_size, num_layers=self.num_lstm_layers, batch_first=True, bidirectional=True).to(self.device)
        self.label_lstm_projection = torch.nn.Linear(in_features=2*self.num_lstm_layers*self.label_representation_size,out_features=self.label_representation_size).to(self.device)

        self.K=10
        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.build()

    def forward(self, x, return_weights=False):
        # Inputs
        embeddings = self.embedding_input_projection(self._embed_inputs(x)/self.embeddings_dim)#.transpose(0,1)
        labels =  self.label_input_projection(self.label_embed(self.label_embeddings)/self.embeddings_dim)
        label_gloves = self.glove_input_projection(self.label_gloves)
        label_sim = label_gloves / label_gloves.norm(dim=-1, p=2, keepdim=True)
        label_sim = torch.softmax((label_sim[None] * label_sim[:, None]).sum(-1), -1)


        embeddings = self.dropout_layer(embeddings)
        labels = self.dropout_layer(labels)

        labels, _ = self.label_transformation(labels.transpose(0,1), labels.transpose(0,1), labels.transpose(0,1))
        labels = self.label_lstm_projection(torch.cat([x for x in self.label_lstm(labels.transpose(0,1))[1][0]],-1))
        labels = self.dropout_layer(labels)
        # labels = self.label_projection(torch.cat([labels, self.label_gloves],-1))
        # Representationst
        labels=torch.matmul(label_sim, labels)


        self_att = self.self_attention(embeddings, label_gloves)
        prior = self.label_attention(embeddings, label_gloves)
        self_att = self.dynamic_fusion(self_att, prior)

        labels = labels[None]
        pred = (labels * self_att).sum(-1) #+ torch.exp(-(labels-self_att).norm(p=2,dim=-1)) #(labels*self_att).sum(-1) ##
        return pred

    def _embed_inputs(self, x):
        if self.finetune:
            if self.n_layers == 1:
                embeddings = self.embedding(x)[0]
            else:
                embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)
        else:
            with torch.no_grad():
                if self.n_layers == 1:
                    embeddings = self.embedding(x)[0]
                else:
                    embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)
        return embeddings

    def label_embed(self, x):
        # Label Embedding Creation
        labels = self._embed_inputs(x)
        return labels

    def self_attention(self, x, labels):
        selfatt = torch.tanh(self.linear_first(x))
        selfatt = torch.matmul(selfatt, self.linear_second(labels).t())
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        return torch.bmm(selfatt, x)

    def label_attention(self,x, labels):
        m1 = torch.matmul(x,labels.t() )
        label_att = torch.relu(torch.bmm(m1.transpose(1,2),x))
        return label_att

    def dynamic_fusion(self,x,y):
        weight1 = torch.sigmoid(self.weight1(x))
        weight2 = torch.sigmoid(self.weight2(y))
        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1
        return weight1 * x + weight2 * y

    def create_label_dict(self):
        if self.explanations is None:
            from ...graph import augment_wikiabstracts, subgraphs, get
            import networkx as nx
            tmp_graph = nx.Graph()
            import re
            tmp_graph.add_nodes_from(
                list(self.classes.keys()) + [w for k in self.classes.keys() for w in re.split("[/_,-]+", k)])
            tmp_graph = augment_wikiabstracts(tmp_graph)

            d = dict(tmp_graph.nodes(True))
            first_level = {w:d.get(w, {}).get("extract", None) for w in self.classes}
            second_level = {w:[d.get(k, {}).get("extract", None) for k in re.split("[/_,-]+", w)] for w in self.classes }

            from mlmc.representation import Embedder
            e = Embedder("glove300")

            def g(x):
                x = x.replace("Ec ","EU ").lower()
                print(x)
                x_embed = e.embed([x])[0].mean(0)[None]
                x_embed /= x_embed.norm(p=2, dim=-1, keepdim=True)
                import requests
                url = f"https://en.wikipedia.org/w/api.php?srlimit=40&format=json&prop=extracts&exintro&explaintext&action=query&list=search&srsearch={x}"
                r = requests.get(url)
                data = r.json()

                result_embeds =  torch.stack([x.mean(0) for x in e.embed([d["title"] for d in data["query"]["search"]])])
                result_embeds /= result_embeds.norm(p=2, dim=-1, keepdim=True)

                sim = (x_embed*result_embeds).sum(-1)
                return data["query"]["search"][sim.argmax()]["snippet"].replace('<span class="searchmatch">', '').replace("</span>","")

            all = {k:v.replace("\n"," ") if not (v is None or "may refer to" in v) else " | ".join([x for x in second_level[k] if x is not None]).replace("\n"," ")  for k,v in first_level.items()}
            self.explanations = {k:v.replace("\n"," ")  if not(v.endswith("may refer to:") or v.endswith("may be:") or "may refer to: " in v or "may refer to:\n" in v or v=="") else g(k).replace("\n"," ")  for k,v in all.items()}

        from ...representation import get_word_embedding_mean
        import re
        with torch.no_grad():
            l = get_word_embedding_mean(
                [" ".join(re.split("[/ _-]", x.lower())) for x in self.classes.keys()],
                "glove300")
        self.glove_dictionary = {w: e for w, e in zip(self.classes, l)}
        self.glove_dim = l.shape[-1]


        transformed = self.tokenizer(["LABEL|| " + k +":"+v for k,v in self.explanations.items()], maxlen=150)
        self.label_embedding_dim = self.embeddings_dim
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

        self.label_embeddings = torch.stack([self.label_dict[cls] for cls in classes.keys()])
        self.label_embeddings = self.label_embeddings.to(self.device)

        self.label_gloves = torch.stack([self.glove_dictionary[cls] for cls in classes.keys()])
        self.label_gloves = self.label_gloves.to(self.device)

    def shuffle(self):
        new_ind = torch.randperm(self.n_classes)
        new_classes = dict(zip(list(self.classes.keys()),new_ind.tolist()))
        self.create_labels(new_classes)