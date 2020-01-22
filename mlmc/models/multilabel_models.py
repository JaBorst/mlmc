import torch
from .abstracts import TextClassificationAbstract
from ..representation import get
##############################################################################################
##############################################################################################
#  Implementations
##############################################################################################


class KimCNN(TextClassificationAbstract):
    def __init__(self, classes, mode="trainable", static=None, transformer=None, max_len=500, **kwargs):
        super(KimCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.max_len = max_len
        self.modes = ("trainable","untrainable","multichannel","transformer")
        self.mode = mode
        self.l = 1
        self.kernel_sizes = [3,4,5,6]
        self.filters = 100
        assert self.mode in self.modes, "%s not in (%s, %s, %s, %s)" % (self.mode, *self.modes)

        if self.mode == "trainable":
            self.embedding_trainable, self.tokenizer = get(static=static, transformer=transformer, freeze=False)
            self.embeddings_dim = self.embedding_trainable.weight.shape[-1]
        elif self.mode == "untrainable":
            self.embedding_untrainable, self.tokenizer = get(static=static, transformer=transformer, freeze=True)
            self.embeddings_dim= self.embedding_untrainable.weight.shape[-1]
        elif self.mode =="multichannel":
            self.l = 2
            self.embedding_untrainable, self.tokenizer = get(static=static,transformer=transformer, freeze=True)
            self.embedding_trainable = torch.nn.Embedding(*self.embedding_untrainable.weight.shape)
            self.embedding_trainable = self.embedding_trainable.from_pretrained(self.embedding_untrainable.weight.clone(),freeze=False)
            self.embeddings_dim = self.embedding_untrainable.weight.shape[-1]
        elif self.mode == "transformer":
            self.transformer, self.tokenizer = get(static=None, transformer=transformer, output_hidden_states=True)
            self.embeddings_dim = torch.cat(self.transformer(self.transformer.dummy_inputs["input_ids"])[2][-5:-1], -1).shape[-1]


        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(self.embeddings_dim, self.filters, k) for k in self.kernel_sizes])
        self.projection = torch.nn.Linear(in_features=self.l*len(self.kernel_sizes)*self.filters, out_features=self.n_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.build()

    def forward(self, x):
        if self.mode == "trainable":
            embedded = self.embedding_trainable(x).permute(0, 2, 1)
            embedded = self.dropout(embedded)
        elif self.mode == "untrainable":
            with torch.no_grad():
                embedded = self.embedding_untrainable(x).permute(0, 2, 1)
            embedded = self.dropout(embedded)
        elif self.mode =="multichannel":
            embedded_1 = self.embedding_trainable(x).permute(0, 2, 1)
            with torch.no_grad():
                embedded_2 = self.embedding_untrainable(x).permute(0, 2, 1)
            embedded_1 = self.dropout(embedded_1)
            embedded_2 = self.dropout(embedded_2)
            embedded = torch.cat([embedded_1,embedded_2], dim=-1)
        elif self.mode == "transformer":
            with torch.no_grad():
                embedded = torch.cat(self.transformer(x)[2][-5:-1], -1).permute(0, 2, 1)
        c = [torch.nn.functional.relu(conv(embedded).permute(0, 2, 1).max(1)[0]) for conv in self.convs]
        output = self.projection(self.dropout(torch.cat(c, 1)))
        return output



class XMLCNN(TextClassificationAbstract):
    def __init__(self, classes, weights, vocabulary, max_len=600, **kwargs):
        super(XMLCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.window = 20
        self.maxlen= 350

        self.embedding_trainable = torch.nn.Embedding(len(vocabulary) + 1, weights.shape[-1])
        self.embedding_trainable.from_pretrained(torch.FloatTensor(weights), freeze=False)
        self.embedding_untrainable = torch.nn.Embedding(len(vocabulary) + 1, weights.shape[-1])
        self.embedding_untrainable.from_pretrained(torch.FloatTensor(weights), freeze=True)

        self.kernels = [3, 4, 5, 6]
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(weights.shape[-1], 100, k) for k in self.kernels ])

        self.dynpool = torch.nn.MaxPool2d(( 1, self.window))
        self.bottleneck = torch.nn.Linear(
            in_features=100*( 2*(sum([(self.maxlen-s+1)// self.window  for s in self.kernels]))),
            out_features=30)
        self.projection = torch.nn.Linear(in_features=30, out_features=self.n_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.sf = torch.nn.Sigmoid()
        self.build()

    def forward(self, x):
        embedded_1 = self.embedding_trainable(x).permute(0, 2, 1)
        embedded_2 = self.embedding_untrainable(x).permute(0, 2, 1)
        embedded_1 = self.dropout(embedded_1)
        embedded_2 = self.dropout(embedded_2)
        c = [torch.nn.functional.relu(self.dynpool(conv(embedded_1))) for conv in self.convs] + \
            [torch.nn.functional.relu(self.dynpool(conv(embedded_2))) for conv in self.convs]
        concat=self.dropout(torch.cat(c,1).view(x.shape[0],-1))
        bn = torch.relu(self.bottleneck(concat))
        output = self.projection(bn)
        return output

    def transform(self, x):
        pd = torch.nn.utils.rnn.pad_sequence([torch.LongTensor(
            [self.vocabulary.get(token.lower(), self.vocabulary["<UNK_TOKEN>"])
             for token in sentence.split(" ")]) for sentence in x],
            batch_first=True, padding_value=0)
        tmp = torch.zeros((len(x), 350), dtype=torch.long)
        tmp[:,:min(350,pd.shape[-1])] = pd[:,:min(350,pd.shape[-1])]
        return tmp


from mlmc.layers.probability import Prob
class KimCNNProb(TextClassificationAbstract):
    def __init__(self, classes, weights, vocabulary, max_len=600, **kwargs):
        super(KimCNNProb, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.vocabulary = vocabulary
        self.max_len = max_len

        self.embedding_trainable = torch.nn.Embedding(len(vocabulary) + 1, weights.shape[-1])
        self.embedding_trainable.from_pretrained(torch.FloatTensor(weights), freeze=False)
        self.embedding_untrainable = torch.nn.Embedding(len(vocabulary) + 1, weights.shape[-1])
        self.embedding_untrainable.from_pretrained(torch.FloatTensor(weights), freeze=True)
        self.convs = torch.nn.ModuleList([torch.nn.Conv1d(weights.shape[-1], 100, k) for k in [3, 4, 5]])
        self.projection = torch.nn.Linear(in_features=600, out_features=self.n_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.prob = Prob(self.n_classes)

        self.build()

    def forward(self, x):
        embedded_1 = self.embedding_trainable(x).permute(0, 2, 1)
        embedded_2 = self.embedding_untrainable(x).permute(0, 2, 1)
        c = [torch.nn.functional.relu(conv(embedded_1).permute(0, 2, 1).max(1)[0]) for conv in self.convs] + \
            [torch.nn.functional.relu(conv(embedded_2).permute(0, 2, 1).max(1)[0]) for conv in self.convs]
        output = self.projection(self.dropout(torch.cat(c, 1)))
        return self.prob(output)

    def transform(self, x):
        return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(
            [self.vocabulary.get(token.lower(), self.vocabulary["<UNK>"])
             for token in sentence.split(" ")]) for sentence in x],
            batch_first=True, padding_value=0)




from mlmc.layers.label_layers import LabelAttention, AdaptiveCombination, LabelSpecificSelfAttention
class LabelSpecificAttention(TextClassificationAbstract):
    """
    Reimplementation of https://github.com/EMNLP2019LSAN/LSAN
    """
    def __init__(self, classes, weights, vocabulary, label_embedding=None, max_len=600,dropout = 0.5, **kwargs):
        super(LabelSpecificAttention, self).__init__(**kwargs)

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

        self.self_attention = LabelSpecificSelfAttention(n_classes=self.n_classes,
                                                         input_dim=self.lstm_units, hidden_dim=200)

        self.label_attention = LabelAttention(self.n_classes, self.lstm_units, hidden_dim=self.lstm_units, label_repr=label_embedding)

        self.adaptive_combination = AdaptiveCombination(2*self.lstm_units, self.n_classes)

        # self.projection_1 = torch.nn.Linear(in_features=2*self.lstm_units, out_features=300)
        # self.projection_2 = torch.nn.Linear(in_features=300, out_features=1)
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
        # combined = torch.relu(self.projection_1(combined))
        # combined = self.projection_2(combined).squeeze(-1)
        return self.projection(combined).squeeze(-1)

    def transform(self, x):
        return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(
            [self.vocabulary.get(token.lower(), self.vocabulary["<UNK_TOKEN>"])
             for token in sentence.split(" ")]) for sentence in x],
            batch_first=True, padding_value=0)



from mlmc.layers.label_layers import LabelEmbeddingScoring
class LabelScoringModel(TextClassificationAbstract):
    """
    Reimplementation of https://github.com/EMNLP2019LSAN/LSAN
    """
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