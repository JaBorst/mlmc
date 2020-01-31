

from mlmc.layers.label_layers import LabelEmbeddingScoring
class LabelScoringModel(TextClassificationAbstract):
    def __init__(self, classes, weights, vocabulary, label_embedding=None, max_len=600,dropout = 0.5, **kwargs):
        super(LabelScoringModel, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.embedding_dim = weights.shape[-1]
        self.lstm_units = self.embedding_dim
        self.use_dropout = dropout

        self.embedding_untrainable = torch.nn.Embedding(weights.shape[0], self.embedding_dim)
        self.embedding_untrainable.from_pretrained(torch.FloatTensor(weights), freeze=True)

        self.lstm = torch.nn.LSTM(self.embedding_dim,
                                  self.lstm_units,
                                  num_layers=1,
                                  bidirectional=True,
                                  batch_first=True)

        self.label_similarity = LabelEmbeddingScoring(self.n_classes,
                                                      2*self.lstm_units,
                                                      label_repr=label_embedding,
                                                      similarity="cosine",
                                                      label_freeze=True)
        self.projection = torch.nn.Linear(in_features=self.n_classes, out_features= self.n_classes)
        if self.use_dropout > 0.0: self.dropout = torch.nn.Dropout(self.use_dropout)
        self.build()

    def forward(self, x):
        embedded_1 = self.embedding_untrainable(x)
        if self.use_dropout > 0.0: embedded_1 = self.dropout(embedded_1)
        c,_ = self.lstm(embedded_1)
        if self.use_dropout > 0.0: c = self.dropout(c)
        ls = self.label_similarity(c).sum(1)
        return self.projection(ls)

    def transform(self, x):
        return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(
            [self.vocabulary.get(token.lower(), self.vocabulary["<UNK_TOKEN>"])
             for token in sentence.split(" ")]) for sentence in x],
            batch_first=True, padding_value=0)

class LabelAttentionScoringModel(TextClassificationAbstract):
    """
    Reimplementation of https://github.com/EMNLP2019LSAN/LSAN
    """
    def __init__(self, classes, weights, vocabulary, label_embedding=None, max_len=600,dropout = 0.5, **kwargs):
        super(LabelAttentionScoringModel, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.embedding_dim = weights.shape[-1]
        self.lstm_units = self.embedding_dim
        self.use_dropout = dropout

        self.embedding_untrainable = torch.nn.Embedding(weights.shape[0], self.embedding_dim)
        self.embedding_untrainable.from_pretrained(torch.FloatTensor(weights), freeze=True)

        self.lstm = torch.nn.LSTM(self.embedding_dim,
                                  self.lstm_units,
                                  num_layers=1,
                                  bidirectional=True,
                                  batch_first=True)

        self.label_similarity = LabelEmbeddingScoring(self.n_classes,
                                                      2*self.lstm_units,
                                                      label_repr=label_embedding,
                                                      similarity="cosine",
                                                      label_freeze=True)
        self.self_attention = LabelSpecificSelfAttention(n_classes=self.n_classes,
                                                         input_dim=self.lstm_units, hidden_dim=200)
        self.projection = torch.nn.Linear(in_features=self.n_classes, out_features= self.n_classes)
        if self.use_dropout > 0.0: self.dropout = torch.nn.Dropout(self.use_dropout)
        self.build()


    def forward(self, x):
        embedded_1 = self.embedding_untrainable(x)
        if self.use_dropout > 0.0: embedded_1 = self.dropout(embedded_1)
        c,_ = self.lstm(embedded_1)
        if self.use_dropout > 0.0: c = self.dropout(c)
        c_view = c.view(c.shape[0], c.shape[1], self.lstm_units,2)
        sc = torch.cat([self.self_attention(c_view[:, :, :, 0])[0], self.self_attention(c_view[:, :, :, 1])[0]], -1)
        ls = self.label_similarity(c).sum(1)
        return self.projection(ls)

    def transform(self, x):
        return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(
            [self.vocabulary.get(token.lower(), self.vocabulary["<UNK_TOKEN>"])
             for token in sentence.split(" ")]) for sentence in x],
            batch_first=True, padding_value=0)