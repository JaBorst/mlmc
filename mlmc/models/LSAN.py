"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from .abstracts import TextClassificationAbstract
from ..representation import get

class LSANOriginal(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, representation, label_embed=None, label_freeze=True, lstm_hid_dim=300, d_a=200 ,max_len=500,**kwargs):
        super(LSANOriginal, self).__init__(**kwargs)
        #My Stuff
        self.max_len = max_len


        # Original
        self.n_classes = len(classes)
        self.embedding, self.tokenizer = get(representation, freeze=True)
        self.embedding_dim = self.embedding(torch.LongTensor([[0]])).shape[-1]

        if label_embed is not None:
            self.label_embed = torch.nn.Embedding(label_embed.shape[0], label_embed.shape[1])
            self.label_embed.from_pretrained(torch.FloatTensor(label_embed), freeze=label_freeze)
        else:
            self.label_embed = torch.nn.Embedding(self.n_classes, self.embedding_dim)

        self.lstm = torch.nn.LSTM(self.embedding_dim, hidden_size=lstm_hid_dim, num_layers=1,
                                  batch_first=True, bidirectional=True)

        self.linear_first = torch.nn.Linear(lstm_hid_dim * 2, d_a)
        self.linear_second = torch.nn.Linear(d_a, self.n_classes)

        self.weight1 = torch.nn.Linear(lstm_hid_dim * 2, 1)
        self.weight2 = torch.nn.Linear(lstm_hid_dim * 2, 1)

        self.output_layer = torch.nn.Linear(lstm_hid_dim * 2, self.n_classes)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)
        self.lstm_hid_dim = lstm_hid_dim
        self.build()

    def init_hidden(self, size):
        return (torch.randn(2, size, self.lstm_hid_dim).to(self.device),
                torch.randn(2, size, self.lstm_hid_dim).to(self.device))

    def forward(self, x):
        embeddings = self.embedding(x)
        embeddings = self.embedding_dropout(embeddings)
        # step1 get LSTM outputs
        # hidden_state = self.init_hidden(x.shape[0])
        outputs, hidden_state = self.lstm(embeddings)#, hidden_state)
        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        # step3 get label-attention
        h1 = outputs[:, :, :self.lstm_hid_dim]
        h2 = outputs[:, :, self.lstm_hid_dim:]

        label = self.embedding_dropout(self.label_embed.weight.data)
        m1 = torch.bmm(label.expand(x.shape[0], self.n_classes, self.lstm_hid_dim), h1.transpose(1, 2))
        m2 = torch.bmm(label.expand(x.shape[0], self.n_classes, self.lstm_hid_dim), h2.transpose(1, 2))
        label_att = torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2)
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
