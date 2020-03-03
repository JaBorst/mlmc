import string

import torch

from .abstracts_lm import LanguageModelAbstract
from ..layers import MogrifierLSTM
from ..representation import map_vocab


class MogrifierLMCharacter(LanguageModelAbstract):
    def __init__(self, alphabet=string.ascii_letters+string.punctuation+"1234567890",
                 hidden_size=128, emb_dim=50, n_layers=1, mogrify_steps=2,
                 learn_initial_states=True, max_len=128, dropout=0.5, inter_dropout=0.2, **kwargs):
        super(MogrifierLMCharacter, self).__init__(**kwargs)


        self.max_len = max_len
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.mogrify_steps = mogrify_steps
        self.learn_initial_states = learn_initial_states
        self.hidden_size = hidden_size
        self.cell_type = MogrifierLSTM
        self.dropout = dropout
        self.inter_dropout = inter_dropout
        self.lm_layers = torch.nn.ModuleList(
            [
                self.cell_type(hidden_size, hidden_size, mogrify_steps)
                if i > 0 else
                self.cell_type(self.emb_dim, hidden_size, mogrify_steps)
                for i in range(n_layers)]
        )
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.inter_dropout_layer = torch.nn.Dropout(inter_dropout)
        self.alphabet = list(alphabet)
        self.alphabet = dict(zip(self.alphabet, range(len(self.alphabet))))
        self.vocabulary_size = len(self.alphabet)
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.emb_dim,
        )
        self.projection = torch.nn.Linear(self.hidden_size, self.vocabulary_size)
        self.build()

    def forward(self, x, representations=False):
        e = self.embedding(x)
        for layer in self.lm_layers:
            e = self.inter_dropout_layer(e)
            e, rep = layer(e)
        if representations:
            return e, rep[0]
        e = self.dropout_layer(e)[:, -1, :]
        return self.projection(e)

    def transform(self, s):
        return torch.tensor([[self.alphabet[x] for x in m] for m in s]).squeeze(-1)

    def encode(self, s):
        return self.transform([x for x in s if x in self.alphabet.keys()]).tolist()

    def decode(self, s):
        alphabet_rev = {v:k for k,v in self.alphabet.items()}
        return "".join([alphabet_rev[x] for x in s])

class MogrifierLMWord(LanguageModelAbstract):
    def __init__(self, word_list, hidden_size=128, emb_dim=50, n_layers=1, mogrify_steps=2,
                 learn_initial_states=True, max_len=128, dropout=0.5, inter_dropout=0.2, **kwargs):
        super(MogrifierLMWord, self).__init__(**kwargs)

        self.max_len = max_len
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.mogrify_steps = mogrify_steps
        self.learn_initial_states = learn_initial_states
        self.hidden_size = hidden_size
        self.cell_type = MogrifierLSTM
        self.lm_layers = torch.nn.ModuleList(
            [
                self.cell_type(hidden_size, hidden_size, mogrify_steps)
                if i > 0 else
                self.cell_type(self.emb_dim, hidden_size, mogrify_steps)
                for i in range(n_layers)]
        )
        self.dropout = dropout
        self.inter_dropout = inter_dropout
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.inter_dropout_layer = torch.nn.Dropout(inter_dropout)
        self.word_list = word_list + ["<UNK_TOKEN>"]
        self.word_list = dict(zip(self.word_list, range(1,len(self.word_list)+1)))
        self.vocabulary_size = len(self.word_list)+1
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.emb_dim,
        )
        self.projection = torch.nn.Linear(self.hidden_size, self.vocabulary_size)
        self.build()

    def forward(self, x, representations=False):
        e = self.embedding(x)
        for layer in self.lm_layers:
            e = self.inter_dropout_layer(e)
            e, rep = layer(e)
        if representations:
            return e, rep[0]
        e = self.dropout_layer(e)[:, -1, :]
        return self.projection(e)

    def transform(self, s):
        if isinstance(s[0], str):
            s = [s]
        return map_vocab(s,self.word_list,len(s[0])).t().squeeze(-1)

    def encode(self, s):
        return self.transform([s.split()]).tolist()

    def decode(self, s):
        return " ".join([list(self.word_list.keys())[x-1] for x in s])
