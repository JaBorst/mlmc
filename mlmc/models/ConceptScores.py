"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
import torch.nn.functional as F
from mlmc.models.abstracts import TextClassificationAbstract
from mlmc.representation import get
import mlmc

class ConceptScores(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, static=None, transformer=None, label_embed=None, label_freeze=True, use_lstm=True, d_a=200, max_len=400, **kwargs):
        super(ConceptScores, self).__init__(**kwargs)
        #My Stuff
        self.classes = classes
        self.max_len = max_len
        self.use_lstm = use_lstm
        self.n_layers = 4

        # Original
        self.n_classes = len(classes)
        self.embedding, self.tokenizer = get(static=static, transformer=transformer, output_hidden_states=True)
        self.embedding_dim = self.embedding(torch.LongTensor([[0]]))[0].shape[-1]*self.n_layers

        self.concepts = torch.nn.Parameter(torch.FloatTensor(label_embed))
        self.concepts_dim = label_embed.shape[-1]
        self.n_concepts = label_embed.shape[0]

        self.concept_projection = torch.nn.Linear(self.embedding_dim, self.concepts_dim)

        self.output_projection = torch.nn.Linear(self.n_concepts, self.n_classes)
        self.build()


    def forward(self, x, return_scores=False):
        with torch.no_grad()  :
            embeddings =  torch.cat(self.embedding(x)[2][(-1-self.n_layers):-1], -1)

        cp = self.concept_projection(embeddings)
        scores = torch.relu(torch.tanh(torch.matmul(cp, self.concepts.permute(1, 0)).mean(-2)))
        output = self.output_projection(scores)
        if return_scores:
            return output, scores
        return output
