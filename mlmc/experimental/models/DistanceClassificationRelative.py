"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from ...models.abstracts import TextClassificationAbstract
from ...models.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ...representation import get
import re
import networkx as nx

class DLCR(TextClassificationAbstract,TextClassificationAbstractZeroShot):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, method="glove", scale="mean", norm=False, representation="roberta", use_lstm=False, d_a=200, max_len=400, n_layers=4, **kwargs):
        super(DLCR, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.representation=representation
        self.method = method
        self.scale = scale
        self.norm = norm
        self.n_layers=n_layers
        self.d_a = d_a

        # Original
        self.n_classes = len(classes)
        self.representation = representation
        self._init_input_representations()

        self.label_dict = self.create_label_dict(method=self.method, scale=self.scale)
        self.create_labels(classes)

        self.input_projection = torch.nn.Linear(self.label_embedding_dim, self.embeddings_dim).to(self.device)
        self.input_projection2 = torch.nn.Linear(self.max_len, self.embeddings_dim).to(self.device)

        self.linear_first = torch.nn.Linear(self.embeddings_dim , self.d_a).to(self.device)
        self.linear_second = torch.nn.Linear(self.embeddings_dim, self.d_a).to(self.device)

        self.weight1 = torch.nn.Linear(self.embeddings_dim, 1).to(self.device)
        self.weight2 = torch.nn.Linear(self.embeddings_dim, 1).to(self.device)

        # self.output_layer = torch.nn.Linear(self.label_embedding_dim * 2, 1).to(self.device)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)

        self.output_layer = torch.nn.Linear(in_features=self.embeddings_dim, out_features=self.d_a)
        self.output_layer2 = torch.nn.Linear(in_features=self.d_a, out_features=1)
        # self.connection = torch.nn.Linear(self.label_embedding_dim * 2, 1)

        self.build()

    def init_hidden(self, size):
        return (torch.randn(2, size, self.label_embedding_dim).to(self.device),
                torch.randn(2, size, self.label_embedding_dim).to(self.device))

    def forward(self, x):
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
        embeddings = self.embedding_dropout(embeddings)

        labels = self.input_projection(self.label_embedding)
        # global_vector = embeddings.mean(-2)

        self_att = torch.sigmoid(self.self_attention(embeddings, labels))
        prior = torch.sigmoid(self.label_attention(embeddings, labels))
        doc = self.dynamic_fusion(self_att, prior)
        doc = self.embedding_dropout(doc)

        pred = self.output_layer2(torch.relu(self.output_layer(doc).squeeze())).squeeze(-1)
        return pred

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

    def create_label_dict(self, method="repeat", scale="mean"):
        # assert method in ("repeat","generate","embed", "glove", "graph"), 'method has to be one of ("repeat","generate","embed")'
        if method == "repeat":
            from ...representation import get_lm_repeated
            with torch.no_grad():
                l = get_lm_repeated(self.classes, self.representation)
        if method == "generate":
            from ...representation import get_lm_generated
            with torch.no_grad():
                l = get_lm_generated(self.classes, self.representation)
        if method == "embed":
            if self.finetune:
                l = self.embedding(self.tokenizer(self.classes.keys()).to(list(self.parameters())[0].device))[1]
            else:
                with torch.no_grad():
                    l = self.embedding(self.tokenizer(self.classes.keys()).to(list(self.parameters())[0].device))[1]
        if method == "glove":
            from ...representation import get_word_embedding_mean
            with torch.no_grad():
                l = get_word_embedding_mean(
                    [" ".join(re.split("[/ _-]", x.lower())) for x in self.classes.keys()],
                    "glove300")
        if method == "wiki":
            from ...graph.graph_operations import augment_wikiabstracts
            tmp_graph = nx.Graph()
            tmp_graph.add_nodes_from(self.classes.keys())
            tmp_graph = augment_wikiabstracts(tmp_graph)
            ls = [dict(val).get("extract", node) for node, val in dict(tmp_graph.nodes(True)).items()]
            with torch.no_grad():
                l = self.embedding(self.tokenizer(ls).to(list(self.parameters())[0].device))[1]

        if scale == "mean":
            print("subtracting mean")
            l = l - l.mean(0, keepdim=True)
        if scale == "normalize":
            print("normalizing")
            l = l / l.norm(p=2, dim=-1, keepdim=True)
        self.label_embedding_dim = l.shape[-1]
        return {w: e for w, e in zip(self.classes, l)}

    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = len(classes)

        if hasattr(self, "label_dict"):
            try:
                self.label_embedding = torch.stack([self.label_dict[cls] for cls in classes.keys()])
            except:
                self.create_label_dict(method=self.method, scale=self.scale)
                self.label_embedding = torch.stack([self.label_dict[cls] for cls in classes.keys()])
        self.label_embedding = self.label_embedding.to(self.device)

