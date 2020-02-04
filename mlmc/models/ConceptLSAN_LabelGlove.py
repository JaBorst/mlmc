"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from .abstracts import TextClassificationAbstract
from ..representation import get, is_transformer


class ConceptLSAN(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, representation="roberta", label_embed=None, label_freeze=True,  d_a=200, max_len=400, **kwargs):
        super(ConceptLSAN, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.n_layers = 4
        self.concept_embedding_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]
        self.representation = representation
        self._init_input_representations()
        # Original
        self.n_classes = len(classes)
        self.label_embed=label_embed
        self.label_freeze = label_freeze
        self.d_a = d_a

        if not is_transformer(self.representation):
            self.lstm = torch.nn.LSTM(self.embedding_dim, self.concept_embedding_dim // 2, 1, bidirectional=True)
        else:
            self.input_projection = torch.nn.Linear(self.embedding_dim, self.concept_embedding_dim)

        self.concept_embedding = torch.nn.Embedding(label_embed.shape[0], label_embed.shape[1])
        self.concept_embedding.from_pretrained(torch.FloatTensor(label_embed), freeze=label_freeze)

        self.linear_first = torch.nn.Linear(self.concept_embedding_dim, d_a)
        self.linear_second = torch.nn.Linear(d_a, self.n_concepts)

        self.weight1 = torch.nn.Linear(self.concept_embedding_dim, 1)
        self.weight2 = torch.nn.Linear(self.concept_embedding_dim, 1)

        self.output_layer = torch.nn.Linear(self.concept_embedding_dim, self.n_classes)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)
        self.connection = torch.nn.Linear(self.n_concepts, self.n_concepts)
        self.build()

    def init_hidden(self, size):
        return (torch.randn(2, size, self.concept_embedding_dim).to(self.device),
                torch.randn(2, size, self.concept_embedding_dim).to(self.device))

    def forward(self, x, return_scores=False):
        with torch.no_grad():
            if is_transformer(self.representation):
                embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
                outputs = self.input_projection(embeddings)
            else:
                embeddings = self.embedding(x)
                embeddings = self.embedding_dropout(embeddings)
                outputs = self.lstm(embeddings)[0]


        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        # step3 get label-attention


        concepts = self.concept_embedding.weight.data
        m1 = torch.bmm(concepts.expand(x.shape[0], *concepts.shape), outputs.transpose(1, 2))
        label_att = torch.relu(torch.bmm(m1, outputs))
        label_att = self.connection(label_att.transpose(-1,-2)).transpose(-1,-2)


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
        avg_sentence_embeddings = torch.sum(doc, 1) / self.n_concepts

        pred = self.output_layer(avg_sentence_embeddings)
        if return_scores:
            return pred, selfatt, m1
        return pred
