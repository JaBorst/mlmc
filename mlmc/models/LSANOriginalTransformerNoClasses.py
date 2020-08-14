"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from .abstracts import TextClassificationAbstract
from .abstracts_zeroshot import TextClassificationAbstractZeroShot
from ..representation import get
import re
import networkx as nx


class LSANOriginalTransformerNoClasses(TextClassificationAbstract,TextClassificationAbstractZeroShot):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, method="glove", scale="mean", norm=False, representation="roberta", use_lstm=False, d_a=200, max_len=400, n_layers=4, **kwargs):
        super(LSANOriginalTransformerNoClasses, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.representation=representation
        self.method = method
        self.scale = scale
        self.norm = norm
        self.n_layers=n_layers

        # Original
        self.n_classes = len(classes)
        self.representation = representation
        self._init_input_representations()

        self.label_dict = self.create_label_dict(method=self.method, scale=self.scale)
        self.create_labels(classes)

        if use_lstm:
            self.lstm = torch.nn.LSTM(self.embeddings_dim, hidden_size=self.label_embedding_dim, num_layers=1,
                                      batch_first=True, bidirectional=True)
        else:
            self.lstm = torch.nn.Linear(self.embeddings_dim, self.label_embedding_dim * 2)

        self.linear_first = torch.nn.Linear(self.label_embedding_dim * 2, d_a)
        self.linear_second = torch.nn.Linear(self.label_embedding_dim, d_a)

        self.weight1 = torch.nn.Linear(self.label_embedding_dim * 2, 1)
        self.weight2 = torch.nn.Linear(self.label_embedding_dim * 2, 1)

        self.output_layer = torch.nn.Linear(self.label_embedding_dim * 2, 1)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)

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
        # step1 get LSTM outputs
        # hidden_state = self.init_hidden(x.shape[0])
        outputs = self.lstm(embeddings)#, hidden_state)
        if self.use_lstm:
            outputs = outputs[0]
        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = torch.matmul(selfatt, self.linear_second(self.label_embedding).t())
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        # step3 get label-attention

        h1 = outputs[:, :, :self.label_embedding_dim]
        h2 = outputs[:, :, self.label_embedding_dim:]

        label = self.label_embedding
        m1 = torch.bmm(label.expand(x.shape[0], *label.shape), h1.transpose(1, 2))
        m2 = torch.bmm(label.expand(x.shape[0], *label.shape), h2.transpose(1, 2))
        label_att = torch.relu(torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2))
        # label_att = self.connection(label_att.transpose(-1,-2)).transpose(-1,-2)


        # label_att = F.normalize(label_att, p=2, dim=-1)
        # self_att = F.normalize(self_att, p=2, dim=-1) #all can
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
        return pred

    def create_label_dict(self, method="repeat", scale="mean"):
        # assert method in ("repeat","generate","embed", "glove", "graph"), 'method has to be one of ("repeat","generate","embed")'
        if method == "repeat":
            from ..representation import get_lm_repeated
            with torch.no_grad():
                l = get_lm_repeated(self.classes, self.representation)
        if method == "generate":
            from ..representation import get_lm_generated
            with torch.no_grad():
                l = get_lm_generated(self.classes, self.representation)
        if method == "embed":
            if self.finetune:
                l = self.embedding(self.tokenizer(self.classes.keys()).to(list(self.parameters())[0].device))[1]
            else:
                with torch.no_grad():
                    l = self.embedding(self.tokenizer(self.classes.keys()).to(list(self.parameters())[0].device))[1]
        if method == "glove":
            from ..representation import get_word_embedding_mean
            with torch.no_grad():
                l = get_word_embedding_mean(
                    [" ".join(re.split("[/ _-]", x.lower())) for x in self.classes.keys()],
                    "glove300")
        if method == "wiki":
            from ..graph.graph_operations import augment_wikiabstracts
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

