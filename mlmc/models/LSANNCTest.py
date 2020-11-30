"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from mlmc.models.abstracts.abstracts import TextClassificationAbstract
from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from ..representation import is_transformer

class LSANNC(TextClassificationAbstract, TextClassificationAbstractZeroShot):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, representation, lstm_hid_dim=300, d_a=200 ,max_len=200,**kwargs):
        super(LSANNC, self).__init__(**kwargs)
        #My Stuff
        self.max_len = max_len


        # Original
        self.classes = classes
        self.n_classes = len(classes)
        self.representation = representation
        self.lstm_hid_dim = lstm_hid_dim
        self._init_input_representations()

        self.create_labels(classes)
        if is_transformer(self.representation):
            self.projection_input = torch.nn.Linear(self.embeddings_dim,
                                                    self.lstm_hid_dim * 2)
        else:
            self.projection_input = torch.nn.LSTM(self.embeddings_dim,
                                                  hidden_size=self.lstm_hid_dim,
                                                  num_layers=1,
                                                  batch_first=True,
                                                  bidirectional=True)

        self.projection_labels = torch.nn.Linear(self.label_embedding.shape[-1], self.lstm_hid_dim )
        self.linear_first = torch.nn.Linear(lstm_hid_dim * 2, d_a)
        self.linear_second = torch.nn.Linear(self.lstm_hid_dim, d_a)

        self.weight1 = torch.nn.Linear(lstm_hid_dim * 2, 1)
        self.weight2 = torch.nn.Linear(lstm_hid_dim * 2, 1)

        self.output_layer = torch.nn.Linear(lstm_hid_dim * 2, 1)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)
        self.lstm_hid_dim = lstm_hid_dim
        self.build()

    def init_hidden(self, size):
        return (torch.randn(2, size, self.lstm_hid_dim).to(self.device),
                torch.randn(2, size, self.lstm_hid_dim).to(self.device))

    def forward(self, x):
        # embeddings = self.embed_input(x) / self.embeddings_dim
        if self.finetune:
            if self.n_layers == 1:
                embeddings = self.embedding(x)[0]
            else:
                embeddings = torch.cat(self.embedding(x)[2][-self.n_layers:], -1)

        embeddings = embeddings / self.embeddings_dim
        label = self.label_embedding

        # step1 get LSTM outputs
        # hidden_state = self.init_hidden(x.shape[0])
        outputs = self.projection_input(embeddings)

        if not is_transformer(self.representation):
            outputs = outputs[0]
        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = torch.matmul(selfatt, self.linear_second(label).t())

        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        # step3 get label-attention
        h1 = outputs[:, :, :self.lstm_hid_dim]
        h2 = outputs[:, :, self.lstm_hid_dim:]

        m1 = torch.bmm(label.expand(x.shape[0],*label.shape), h1.transpose(1, 2))
        m2 = torch.bmm(label.expand(x.shape[0], *label.shape), h2.transpose(1, 2))
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

        pred = self.output_layer(doc).squeeze(-1)
        return pred


    def label_embed(self, classes):
        from ..representation import get_word_embedding_mean
        import re
        with torch.no_grad():
            l = get_word_embedding_mean(
                [" ".join(re.split("[/ _-]", x.lower())) for x in self.classes.keys()],
                "glove300")
        return l
