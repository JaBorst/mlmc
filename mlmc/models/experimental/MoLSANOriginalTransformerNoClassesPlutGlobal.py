"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from ..abstracts_mo import TextClassificationAbstractMultiOutput
import re
import networkx as nx


class MoLSANOriginalTransformerNoClassesGlobal(TextClassificationAbstractMultiOutput):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, method="glove", scale="nothing", norm=False, representation="roberta",  dropout=0.5, d_a=400, max_len=400, n_layers=4, **kwargs):
        super(MoLSANOriginalTransformerNoClassesGlobal, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.dropout = dropout
        self.representation=representation
        self.method = method
        self.scale = scale
        self.norm = norm
        self.n_layers=n_layers
        self.kernel_sizes = [3, 4, 5, 6]
        self.filters = 100

        # Original
        self.n_classes = [len(x) for x in self.classes]
        self.representation = representation
        self._init_input_representations()

        self.create_labels(classes)


        self.input_projection = torch.nn.ModuleList([torch.nn.Linear(self.embeddings_dim, self.label_embedding_dim * 2) for _ in range(len(self.n_classes))])

        self.linear_first = torch.nn.ModuleList([torch.nn.Linear(self.label_embedding_dim * 2, d_a) for _ in range(len(self.n_classes))])
        self.linear_second =torch.nn.ModuleList([torch.nn.Linear(d_a, self.n_classes[cls]) for cls in range(len(self.n_classes))]).to(self.device)

        self.weight1 = torch.nn.ModuleList([torch.nn.Linear(self.label_embedding_dim * 2, 1) for _ in range(len(self.n_classes))])
        self.weight2 = torch.nn.ModuleList([torch.nn.Linear(self.label_embedding_dim * 2, 1) for _ in range(len(self.n_classes))])
        self.weight3 = torch.nn.ModuleList([torch.nn.Linear(self.label_embedding_dim * 2, 1) for _ in range(len(self.n_classes))]).to(self.device)

        # self.output_layer = torch.nn.ModuleList([torch.nn.Linear(self.label_embedding_dim * 2, 1) for _ in self.n_classes]).to(self.device)
        self.output_layer = torch.nn.Linear(self.label_embedding_dim * 2, 1)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)

        # self.connection = torch.nn.Linear(self.label_embedding_dim * 2, 1)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.embeddings_dim, self.filters, k) for k in self.kernel_sizes])
        self.global_projection = torch.nn.Linear(len(self.kernel_sizes)*self.filters, self.label_embedding_dim * 2).to(self.device)
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
        global_vector = [torch.nn.functional.relu(conv(embeddings.transpose(1,2)).permute(0, 2, 1).max(1)[0]) for conv in self.convs]
        global_vector = self.global_projection(torch.cat(global_vector, 1))
        outputs = [i(embeddings) for i in self.input_projection]
        doc = [self.docs(o,global_vector,i) for o,i in zip(outputs,range(len(self.n_classes)))]
        # pred = [o(d).squeeze() for o,d in zip(self.output_layer,doc)]
        pred = [self.output_layer(d).squeeze() for d in doc]

        return pred

    def label_spec_doc_rep(self, x, i):
        # l = self.label_embedding[i]
        selfatt = torch.tanh(self.linear_first[i](x))
        selfatt = F.softmax( self.linear_second[i](selfatt), dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, x)
        return self_att

    def label_spec_label_rep(self, x, i):
        l = self.label_embedding[i]
        h1 = x[:, :, :self.label_embedding_dim]
        h2 = x[:, :, self.label_embedding_dim:]
        m1 = torch.bmm(l.expand(x.shape[0], *l.shape), h1.transpose(1, 2))
        m2 = torch.bmm(l.expand(x.shape[0], *l.shape), h2.transpose(1, 2))
        label_att = torch.relu(torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2))
        return label_att

    def adaptive_fusion(self,x,y,z,i):
        weight1 = torch.sigmoid(self.weight1[i](x))
        weight2 = torch.sigmoid(self.weight2[i](y))
        weight3 = torch.sigmoid(self.weight3[i](z))[:,None,:]
        weight1 = weight1 / (weight1 + weight2 + weight3)
        weight3 = weight3 / (weight1 + weight2 + weight3)
        weight2 = 1 - weight1

        doc = weight1 * x + weight2 * y + weight3*z[:,None,:]
        doc = self.embedding_dropout(doc)/ self.label_embedding_dim
        return doc

    def docs(self, outputs, global_vector, i):
        return self.adaptive_fusion(self.label_spec_doc_rep(outputs,i), self.label_spec_label_rep(outputs, i),global_vector,i)


    def create_label_dict(self):
        # assert method in ("repeat","generate","embed", "glove", "graph"), 'method has to be one of ("repeat","generate","embed")'
        from ...representation import get_word_embedding_mean
        with torch.no_grad():
            l = [get_word_embedding_mean(
                [(" ".join(re.split("[/ _-]", re.sub("[0-9]","",x.lower())))).strip() for x in classes.keys()],
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

        if not hasattr(self, "label_dict"):
            self.label_dict = self.create_label_dict()
        self.label_embedding = [torch.stack([dic[cls] for cls in cls.keys()]) for cls, dic in
                                zip(self.classes, self.label_dict)]
        self.label_embedding = torch.nn.ParameterList([torch.nn.Parameter(x)for x in self.label_embedding]).to(self.device)
        for x in self.label_embedding: x.requires_grad=True
