"""
Few-Shot and Zero-Shot Multi-Label Learning for Structured Label Spaces - Rios & Kavuluru (2018)
"""
import torch
from .abstracts import TextClassificationAbstract

class ZAGCNN(TextClassificationAbstract):
    def __init__(self, weights, classes, vocabulary,  label_embedding=None, max_len=600, dropout = 0.5, **kwargs):
        super(ZAGCNN, self).__init__(**kwargs)

        self.classes = classes
        self.n_classes = len(classes)
        self.vocabulary = vocabulary
        self.max_len = max_len
        self.embedding_dim = weights.shape[-1]
        self.lstm_units = self.embedding_dim
        self.use_dropout = dropout


        self.embedding_untrainable = torch.nn.Embedding(weights.shape[0], self.embedding_dim)
        self.embedding_untrainable.from_pretrained(torch.FloatTensor(weights), freeze=True)

    def transform(self, x):
        return torch.nn.utils.rnn.pad_sequence([torch.LongTensor(
            [self.vocabulary.get(token.lower(), self.vocabulary["<UNK_TOKEN>"])
                                                               for token in sentence.split(" ")]) for sentence in x],
                                                             batch_first=True, padding_value=0)
