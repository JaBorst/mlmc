import torch
from ..layers import MogrifierLSTM, MogLSTM
from ..layers import LSTM
from .abstracts_lm import LanguageModelAbstract

class MogrifierLM(LanguageModelAbstract):
    def __init__(self, hidden_size, embedding_dim, n_layers, mogrify_steps,learn_initial_states=True, max_len=128,dropout=0.5, inter_dropout=0.2, **kwargs):
        super(MogrifierLM, self).__init__(**kwargs)
        self.max_len=max_len
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.mogrify_steps = mogrify_steps
        self.learn_initial_states = learn_initial_states
        self.hidden_size = hidden_size
        self.cell_type = MogLSTM
        self.lm_layers=torch.nn.ModuleList(
            [
                self.cell_type(hidden_size,hidden_size,mogrify_steps)
                if i >0 else
                self.cell_type(self.embedding_dim,hidden_size,mogrify_steps)
             for i in range(n_layers)]
        )
        # self.lm_layers = torch.nn.ModuleList(
        #     [
        #         torch.nn.GRU(hidden_size, hidden_size, mogrify_steps, batch_first=True)
        #         if i > 0 else
        #         torch.nn.GRU(self.embedding_dim, hidden_size, mogrify_steps, batch_first=True)
        #         for i in range(n_layers)]
        # )
        self.dropout = torch.nn.Dropout(dropout)
        self.inter_dropout = torch.nn.Dropout(inter_dropout)
        self.build()

    def forward(self, x, representations=False):
        e = self.embedding(x)
        for layer in self.lm_layers:
            e = self.inter_dropout(e)
            e, rep = layer(e)
        if representations:
            return e, rep[0]
        e = self.dropout(e)[:,-1,:]
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
        self.alphabet = list(alphabet)
        self.alphabet = dict(zip(self.alphabet, range(len(self.alphabet))))
        self.vocabulary_size = len(self.alphabet)
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.embedding_dim,
        )
        self.projection = torch.nn.Linear(self.hidden_size, self.vocabulary_size)
        self.build()

    def transform(self, s):
        return torch.tensor([[self.alphabet[x] for x in m] for m in s]).squeeze(-1)

    def encode(self, s):
        return self.transform([x for x in s if x in self.alphabet.keys()]).tolist()

    def decode(self, s):
        return "".join([list(self.alphabet.keys())[x] for x in s])

class MogrifierLMWord(MogrifierLM):
    def __init__(self, word_list, **kwargs):
        super(MogrifierLMWord, self).__init__(**kwargs)
        self.word_list = word_list + ["<UNK_TOKEN>"]
        self.word_list = dict(zip(self.word_list, range(1,len(self.word_list)+1)))
        self.vocabulary_size = len(self.word_list)+1
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.embedding_dim,
        )
        self.projection = torch.nn.Linear(self.hidden_size, self.vocabulary_size)
        self.build()

    def transform(self, s):
        if isinstance(s[0], str):
            s = [s]
        return map_vocab(s,self.word_list,len(s[0])).t().squeeze(-1)

    def encode(self, s):
        return self.transform([s.split()]).tolist()

    def decode(self, s):
        return " ".join([list(self.word_list.keys())[x-1] for x in s])

class MogrifierLMTokenizer(MogrifierLM):
    def __init__(self, tokenizer, **kwargs):
        super(MogrifierLMTokenizer, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.vocabulary_size = tokenizer.get_vocab_size()
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=self.embedding_dim
        )
        self.projection = torch.nn.Linear(self.hidden_size, self.vocabulary_size)
        self.build()
