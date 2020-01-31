import torch
from ..layers.label_layers import LabelAttention, AdaptiveCombination, LabelSpecificSelfAttention
from ..models.abstracts import TextClassificationAbstract
from ..representation import get_embedding

class LabelSpecificAttention(TextClassificationAbstract):
    """
    Reimplementation of https://github.com/EMNLP2019LSAN/LSAN
    """
    def __init__(self, classes, representation, label_embedding=None, max_len=600,dropout = 0.5, **kwargs):
        super(LabelSpecificAttention, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.lstm_units = self.embedding_dim
        self.use_dropout = dropout

        self.embedding_trainable, self.tokenizer = get_embedding(representation, freeze=False)
        self.embedding_dim = self.embedding(torch.LongTensor([[0]])).shape[-1]


        self.lstm = torch.nn.LSTM(self.embedding_dim,
                                  self.lstm_units,
                                  num_layers=1,
                                  bidirectional=True,
                                  batch_first=True)

        self.self_attention = LabelSpecificSelfAttention(n_classes=self.n_classes,
                                                         input_dim=self.lstm_units, hidden_dim=200)

        self.label_attention = LabelAttention(self.n_classes, self.lstm_units, hidden_dim=self.lstm_units, label_repr=label_embedding)
        self.adaptive_combination = AdaptiveCombination(2*self.lstm_units, self.n_classes)
        self.projection = torch.nn.Linear(in_features=2*self.lstm_units, out_features= 1)
        if self.use_dropout > 0.0: self.dropout = torch.nn.Dropout()
        self.build()

    def forward(self, x):
        embedded_1 = self.embedding_untrainable(x)#.permute(0, 2, 1)
        if self.use_dropout > 0.0: embedded_1 = self.dropout(embedded_1)
        c,_ = self.lstm(embedded_1)
        c = c.view(c.shape[0], c.shape[1], self.lstm_units,2)
        if self.use_dropout > 0.0: c = self.dropout(c)

        sc = torch.cat([self.self_attention(c[:,:,:,0])[0],self.self_attention(c[:,:,:,1])[0]],-1)
        la = torch.cat([self.label_attention(c[:,:,:,0])[0],self.label_attention(c[:,:,:,1])[0]],-1)
        combined = self.adaptive_combination([sc, la])
        if self.use_dropout > 0.0: self.dropout(combined)
        return self.projection(combined).squeeze(-1)
