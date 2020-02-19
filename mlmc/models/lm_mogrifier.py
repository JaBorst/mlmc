import torch
from ..layers import MogrifierLSTM
from .abstracts_lm import LanguageModelAbstract

class MogrifierLM(LanguageModelAbstract):
    def __init__(self, hidden_size, n_layers, mogrify_steps, max_len=128,dropout=0.5, **kwargs):
        super(MogrifierLM, self).__init__(**kwargs)
        self.max_len=max_len
        self.embedding_dim = hidden_size
        self.n_layers = n_layers
        self.mogrify_steps = mogrify_steps
        self.hidden_size = hidden_size


        self.lm_layers=torch.nn.ModuleList([MogrifierLSTM(hidden_size,hidden_size,mogrify_steps) for _ in range(n_layers)])


        self.dropout = torch.nn.Dropout(dropout)
        self.build()

    def forward(self, x, representations=False):
        e = self.embedding(x)
        for layer in self.lm_layers:
            e = self.dropout(e)
            e, rep = layer(e)
        e = self.dropout(e)
        if representations:
            return rep[0]
        return self.projection(e)

    def representations(self, s):
        if not isinstance(s, list):
            s = [s]
        return self(self.transform([e.tokens for e in self.tokenizer.encode_batch(s)]).to(self.device), representations=True)

import string
from ..representation import map_vocab
class MogrifierLMCharacter(MogrifierLM):
    def __init__(self, alphabet=string.ascii_letters+string.punctuation+"1234567890", **kwargs):
        super(MogrifierLMCharacter, self).__init__(**kwargs)
        self.alphabet = list(alphabet) + ["<UNK_TOKEN>"]
        self.alphabet = dict(zip(self.alphabet, range(1,len(self.alphabet)+1)))
        self.vocabulary_size = len(self.alphabet)+1
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.hidden_size,
        )
        self.projection = torch.nn.Linear(self.hidden_size, self.vocabulary_size)
        self.build()

    def transform(self, s):
        if not isinstance(s, list):
            s = [s]
        return map_vocab(s,self.alphabet,len(s[0])).t()

class MogrifierLMWord(MogrifierLM):
    def __init__(self, word_list, **kwargs):
        super(MogrifierLMWord, self).__init__(**kwargs)
        self.word_list = word_list + ["<UNK_TOKEN>"]
        self.word_list = dict(zip(self.word_list, range(1,len(self.word_list)+1)))
        self.vocabulary_size = len(self.word_list)+1
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.hidden_size,
        )
        self.projection = torch.nn.Linear(self.hidden_size, self.vocabulary_size)
        self.build()

    def transform(self, s):
        if not isinstance(s, list):
            s = [s]
        return map_vocab(s,self.word_list,len(s[0])).t()

class MogrifierLMTokenizer(MogrifierLM):
    def __init__(self, tokenizer, **kwargs):
        super(MogrifierLMTokenizer, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.vocabulary_size = tokenizer.get_vocab_size()
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.hidden_size
        )
        self.projection = torch.nn.Linear(self.hidden_size, self.vocabulary_size)
        self.build()
