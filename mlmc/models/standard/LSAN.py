"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from ..abstracts.abstract_textclassification import TextClassificationAbstract
from ...representation import is_transformer

class LSAN(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, label_embedding=None, label_freeze=False, lstm_hid_dim=300, d_a=200,  **kwargs):
        """
        Class constructor and initialization of every hyperparameter.

        :param label_embedding: If not None, loads the specified label embeddings instead of creating them from scratch.
        :param label_freeze: If True, the embedding tensor does not get updated in the learning process.
        :param lstm_hid_dim: Hidden state dimension of the LSTM used to create the word embeddings
        :param d_a: Arbitrarily set hyperparameter
        :param kwargs: Optimizer and loss function keyword arguments, see `mlmc.models.abstracts.abstracts.TextClassificationAbstract`
        """
        super(LSAN, self).__init__(**kwargs)
        #My Stuff
        # self.label_embedding = label_embedding
        # self.label_freeze = label_freeze
        # self.d_a = d_a
        # self.lstm_hid_dim = lstm_hid_dim

        self._config["lstm_hid_dim"] = lstm_hid_dim
        self._config["d_a"] = d_a
        self._config["label_freeze"] = label_freeze
        self._config["label_embedding"] = label_embedding


        if self._config["label_embedding"] is not None:
            self.label_embed = torch.nn.Embedding(
                self._config["label_embedding"].shape[0],
                self._config["label_embedding"].shape[1])
            self.label_embed.from_pretrained(torch.FloatTensor(self._config["label_embedding"]), freeze=label_freeze)
        else:
            self.label_embed = torch.nn.Embedding(self.n_classes, self._config["lstm_hid_dim"])

        if is_transformer(self.representation):
            self.projection_input = torch.nn.Linear(self.embeddings_dim,
                                                    self._config["lstm_hid_dim"] * 2)
        else:
            self.projection_input = torch.nn.LSTM(self.embeddings_dim,
                                                  hidden_size=self._config["lstm_hid_dim"],
                                                  num_layers=1,
                                                  batch_first=True,
                                                  bidirectional=True)

        self.linear_first = torch.nn.Linear(self._config["lstm_hid_dim"] * 2, d_a)
        self.linear_second = torch.nn.Linear(self._config["d_a"], self.n_classes)

        self.weight1 = torch.nn.Linear(self._config["lstm_hid_dim"] * 2, 1)
        self.weight2 = torch.nn.Linear(self._config["lstm_hid_dim"] * 2, 1)

        self.output_layer = torch.nn.Linear(self._config["lstm_hid_dim"] * 2, self.n_classes)
        self.embedding_dropout = torch.nn.Dropout(p=0.5)
        self.build()

    def init_hidden(self, size):
        """Deprecated"""
        return (torch.randn(2, size, self._config["lstm_hid_dim"]).to(self.device),
                torch.randn(2, size, self._config["lstm_hid_dim"]).to(self.device))

    def forward(self, x):
        """
        Forward pass function for transforming input tensor into output tensor.

        :param x: Input tensor
        :return: Output tensor
        """
        embeddings = self.embed_input(x) / self.embeddings_dim
        embeddings = self.embedding_dropout(embeddings)
        # step1 get LSTM outputs
        # hidden_state = self.init_hidden(x.shape[0])
        outputs = self.projection_input(embeddings)
        if not is_transformer(self._config["representation"]):
            outputs = outputs[0]
        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        # step3 get label-attention
        h1 = outputs[:, :, :self._config["lstm_hid_dim"]]
        h2 = outputs[:, :, self._config["lstm_hid_dim"]:]

        label = self.embedding_dropout(self.label_embed.weight.data)
        m1 = torch.bmm(label.expand(embeddings.shape[0], self.n_classes, self._config["lstm_hid_dim"]), h1.transpose(1, 2))
        m2 = torch.bmm(label.expand(embeddings.shape[0], self.n_classes, self._config["lstm_hid_dim"]), h2.transpose(1, 2))
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
