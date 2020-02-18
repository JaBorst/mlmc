import torch
from ..layers import MogrifierLSTM
from .abstracts_lm import LanguageModelAbstract

class MogrifierLM(LanguageModelAbstract):
    def __init__(self, hidden_size, n_layers, mogrify_steps, tokenizer, embedding_dim, dropout=0.5, **kwargs):
        super(MogrifierLM, self).__init__(**kwargs)
        self.max_len=256
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.mogrify_steps = mogrify_steps
        self.tokenizer = tokenizer
        self.vocabulary_size = tokenizer.get_vocab_size()

        self.embedding = torch.nn.Embedding(
            num_embeddings=self.vocabulary_size,
            embedding_dim=hidden_size,
            # padding_index=self.padding_idx,
        )

        self.lm_layers=torch.nn.ModuleList([MogrifierLSTM(hidden_size,hidden_size,mogrify_steps) for _ in range(n_layers)])
        self.projection=torch.nn.Linear(hidden_size,self.vocabulary_size)

        self.dropout = torch.nn.Dropout(dropout)
        self.build()

    def forward(self, x, representations=False):
        e = self.embedding(x)
        for layer in self.lm_layers:
            e = self.dropout(e)
            e, rep = layer(e)
        if representations:
            return rep[0]
        return self.projection(e)

    def transform(self, x):
        return torch.LongTensor([[self.tokenizer.token_to_id(t) for t in sequence] for sequence in x]).t()

    def representations(self, s):
        if not isinstance(s, list):
            s = [s]
        return self(self.transform([e.tokens for e in self.tokenizer.encode_batch(s)]).to(self.device), representations=True)