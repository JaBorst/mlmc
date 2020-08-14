"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from ..abstracts_mo import TextClassificationAbstractMultiOutput
import re
import networkx as nx


class MoLSANOriginalTransformerNoClasses2(TextClassificationAbstractMultiOutput):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, method="glove", scale="nothing", norm=False, representation="roberta",  dropout=0.5, d_a=200, max_len=400, n_layers=4, **kwargs):
        super(MoLSANOriginalTransformerNoClasses2, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.dropout = dropout
        self.representation=representation
        self.method = method
        self.scale = scale
        self.norm = norm
        self.n_layers=n_layers

        # Original
        self.n_classes = [len(x) for x in self.classes]
        self.representation = representation
        self._init_input_representations()

        self.create_labels(classes)


        self.input_projection = torch.nn.Linear(self.embeddings_dim, self.label_embedding_dim * 2)

        self.linear_first = torch.nn.Linear(self.label_embedding_dim * 2, d_a)
        self.linear_second = torch.nn.Linear(self.label_embedding_dim, d_a)

        self.weight1 = torch.nn.Linear(self.label_embedding_dim * 2, 1)
        self.weight2 = torch.nn.Linear(self.label_embedding_dim * 2, 1)

        self.output_layer = torch.nn.ModuleList([torch.nn.Linear(self.label_embedding_dim * 2, 1) for _ in self.n_classes]).to(self.device)
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
        else:
            with torch.no_grad():
                if self.n_layers == 1:
                    embeddings = self.embedding(x)[0]
                else:
                    embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)

        embeddings = self.embedding_dropout(embeddings)
        outputs = self.input_projection(embeddings)
        labels = torch.cat(self.label_embedding)
        doc = self.adaptive_fusion(self.label_spec_doc_rep(outputs, labels), self.label_spec_label_rep(outputs, labels))
        doc = [doc[:,self.index[l]:self.index[l+1]] for l in range(len(self.index) -1)]
        pred = [o(d).squeeze() for o,d in zip(self.output_layer,doc)]
        return pred

    def label_spec_doc_rep(self, x, l):
        selfatt = torch.tanh(self.linear_first(x))
        selfatt = torch.matmul(selfatt, self.linear_second(l).t())
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, x)
        return self_att

    def label_spec_label_rep(self, x, l):
        h1 = x[:, :, :self.label_embedding_dim]
        h2 = x[:, :, self.label_embedding_dim:]
        m1 = torch.bmm(l.expand(x.shape[0], *l.shape), h1.transpose(1, 2))
        m2 = torch.bmm(l.expand(x.shape[0], *l.shape), h2.transpose(1, 2))
        label_att = torch.relu(torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2))
        return label_att

    def adaptive_fusion(self,x,y):
        weight1 = torch.sigmoid(self.weight1(x))
        weight2 = torch.sigmoid(self.weight2(y))
        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1

        doc = weight1 * x + weight2 * y
        doc = self.embedding_dropout(doc)/ self.label_embedding_dim
        return doc

    def create_label_dict(self):
        # assert method in ("repeat","generate","embed", "glove", "graph"), 'method has to be one of ("repeat","generate","embed")'
        from ...representation import get_word_embedding_mean
        with torch.no_grad():
            l = [get_word_embedding_mean(
                [" ".join(re.split("[/ _-]", x.lower())) for x in classes.keys()],
                "glove300") for classes in self.classes]

        if self.scale == "mean":
            print("subtracting mean")
            for i in range(len(l)):
                l[i] = l[i] - l[i].mean(0, keepdim=True)
        if self.scale == "normalize":
            print("normalizing")
            for i in range(len(l)):
                l[i] = l[i] / l[i].norm(p=2, dim=-1, keepdim=True)
        self.label_embedding_dim = l[0].shape[-1]
        return [{w: e for w, e in zip(classes.keys(), emb)} for classes, emb in zip(self.classes, l)]

    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = [len(x) for x in classes]
        self.index = self.n_classes
        self.index = [sum(self.index[:l]) for l in range(len(self.index) + 1)]

        if not hasattr(self, "label_dict"):
            self.label_dict = self.create_label_dict()
        self.label_embedding = [torch.stack([dic[cls] for cls in cls.keys()]) for cls, dic in
                                zip(self.classes, self.label_dict)]
        self.label_embedding = [x.to(self.device) for x in self.label_embedding]

