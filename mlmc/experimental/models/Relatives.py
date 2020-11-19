"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from mlmc.models.abstracts.abstracts_graph import TextClassificationAbstractGraph
from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
import re


class LSANR(TextClassificationAbstractGraph, TextClassificationAbstractZeroShot):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, norm=False, representation="roberta", use_lstm=False, d_a=200, max_len=400, n_layers=4, **kwargs):
        super(LSANR, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.representation=representation
        self.norm = norm
        self.n_layers=n_layers
        self.d_a = d_a

        # Original
        self.n_classes = len(classes)
        self.representation = representation
        self._init_input_representations()

        self.create_labels(classes)




        self.input_projection = torch.nn.Linear(self.label_embeddings_dim, self.embeddings_dim).to(self.device)

        self.linear_first = torch.nn.Linear(self.embeddings_dim , self.d_a).to(self.device)
        self.linear_second = torch.nn.Linear(self.embeddings_dim, self.d_a).to(self.device)

        self.weight1 = torch.nn.Linear(self.embeddings_dim, 1).to(self.device)
        self.weight2 = torch.nn.Linear(self.embeddings_dim, 1).to(self.device)

        # self.output_layer = torch.nn.Linear(self.label_embeddings_dim * 2, 1).to(self.device)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)



        self.output_layer = torch.nn.Linear(in_features=self.embeddings_dim, out_features=200)
        self.output_layer2 = torch.nn.Linear(in_features=200, out_features=1)
        # self.connection = torch.nn.Linear(self.label_embeddings_dim * 2, 1)
        # import torch_geometric as torchg
        # self.gcn = torchg.nn.GraphConv(in_channels=self.label_embeddings_dim,
        #                                out_channels=self.label_embeddings_dim).to(self.device)
        # self.gcn2 = torchg.nn.GraphConv(in_channels=self.label_embeddings_dim,
        #                                 out_channels=self.label_embeddings_dim).to(self.device)
        # self.gcn = torchg.nn.GatedGraphConv( out_channels=self.embeddings_dim,  num_layers=self.depth+2).to(self.device)

        self.build()

    def forward(self, x, return_scores=False):

        # Inputs
        embeddings = self._embed_inputs(x)
        embeddings = self.embedding_dropout(embeddings)
        labels = self.input_projection(self.label_embeddings)


        # Representations
        self_att = torch.sigmoid(self.self_attention(embeddings, labels))
        prior = torch.sigmoid(self.label_attention(embeddings, labels))
        doc = self.dynamic_fusion(self_att, prior)
        doc = self.embedding_dropout(doc)

        self.position = torch.softmax(torch.diagonal(((doc[:, :, None] - labels[None, None])**2).sum(-1),dim1=1, dim2=2),-1)
        pred = self.output_layer2(torch.tanh(torch.relu(self.output_layer(doc).squeeze()))).squeeze(-1)
        if return_scores:
            return pred
        return pred[:,:self.n_classes]

    def regularize(self):
        return self.position.mean()

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

        from ...representation import get_word_embedding_mean
        with torch.no_grad():
            l = get_word_embedding_mean(
                [" ".join(re.split("[/ _-]", x.lower())) for x in self.classes.keys()],
                "glove300")

        self.label_embeddings_dim = l.shape[-1]
        return {w: e for w, e in zip(self.classes, l)}

    def create_labels(self, classes):
        self.classes = classes
        self.n_classes = len(classes)

        if not hasattr(self, "label_dict"):
            self.label_dict = self.create_label_dict()

        self.label_embeddings = torch.stack([self.label_dict[cls] for cls in classes.keys()])
        self.label_embeddings = self.label_embeddings.to(self.device)

