"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from .abstracts import TextClassificationAbstract
from ..representation import get


class LSANOriginalTransformer(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, representation=None, label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(LSANOriginalTransformer, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = 4
        self.representation=representation

        # Original
        self.n_classes = len(classes)
        self.embedding, self.tokenizer = get(representation, output_hidden_states=True)
        self.embedding_dim = self.embedding(torch.LongTensor([[0]]))[0].shape[-1]*self.n_layers

        if label_embed is not None:
            self.label_embedding = torch.nn.Embedding(label_embed.shape[0], label_embed.shape[1])
            self.label_embedding.from_pretrained(torch.FloatTensor(label_embed), freeze=label_freeze)
        else:
            self.label_embedding = torch.nn.Embedding(label_embed.shape[0], self.embedding_dim)

        self.label_embedding_dim = label_embed.shape[-1]

        if use_lstm:
            self.lstm = torch.nn.LSTM(self.embedding_dim, hidden_size=self.label_embedding_dim, num_layers=1,
                                      batch_first=True, bidirectional=True)
        else:
            self.lstm = torch.nn.Linear(self.embedding_dim, self.label_embedding_dim* 2)

        self.linear_first = torch.nn.Linear(self.label_embedding_dim * 2, d_a)
        self.linear_second = torch.nn.Linear(d_a, self.n_classes)

        self.weight1 = torch.nn.Linear(self.label_embedding_dim * 2, 1)
        self.weight2 = torch.nn.Linear(self.label_embedding_dim * 2, 1)

        self.output_layer = torch.nn.Linear(self.label_embedding_dim * 2, self.n_classes)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)

        self.connection = torch.nn.Linear(label_embed.shape[0], self.n_classes)

        self.build()

    def init_hidden(self, size):
        return (torch.randn(2, size, self.label_embedding_dim).to(self.device),
                torch.randn(2, size, self.label_embedding_dim).to(self.device))

    def forward(self, x):
        with torch.no_grad()  :
            embeddings =  torch.cat(self.embedding(x)[2][(-1-self.n_layers):-1], -1)

        embeddings = self.embedding_dropout(embeddings)
        # step1 get LSTM outputs
        # hidden_state = self.init_hidden(x.shape[0])
        outputs = self.lstm(embeddings)#, hidden_state)
        if self.use_lstm:
            outputs = outputs[0]
        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        # step3 get label-attention

        h1 = outputs[:, :, :self.label_embedding_dim]
        h2 = outputs[:, :, self.label_embedding_dim:]

        label = self.label_embedding.weight.data
        m1 = torch.bmm(label.expand(x.shape[0], *label.shape), h1.transpose(1, 2))
        m2 = torch.bmm(label.expand(x.shape[0], *label.shape), h2.transpose(1, 2))
        label_att = torch.relu(torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2))
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
        avg_sentence_embeddings = torch.sum(doc, 1) / self.n_classes

        pred = self.output_layer(avg_sentence_embeddings)
        return pred
