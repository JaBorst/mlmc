"""
https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
"""
import torch
from .abstracts import TextClassificationAbstract
from ..representation import get, is_transformer
import re
from ..representation.labels import makemultilabels


class GloveConcepts(TextClassificationAbstract):
    """
    https://raw.githubusercontent.com/EMNLP2019LSAN/LSAN/master/attention/model.py
    """
    def __init__(self, classes, concepts, label_vocabulary, representation="roberta", label_freeze=True, max_len=400, **kwargs):
        super(GloveConcepts, self).__init__(**kwargs)
        #My Stuff
        assert is_transformer(representation), "This model only works with transformers"

        self.classes = classes
        self.max_len = max_len
        self.n_layers = 4
        self.concept_embedding_dim = concepts.shape[-1]
        self.n_concepts = concepts.shape[0]
        self.representation = representation
        self._init_input_representations()
        # Original
        self.n_classes = len(classes)
        self.label_freeze = label_freeze
        self.label_vocabulary = label_vocabulary
        self.att_dim = 256

        self.concepts=torch.nn.Parameter(torch.from_numpy(concepts).float())
        self.concepts.requires_grad=False

        label_embed = [[self.label_vocabulary.get(w,0) if w != "comdedy" else self.label_vocabulary["comedy"] for w in re.split("[ ,'-]",x.lower())] for x in self.classes.keys()]
        # label_embed = [torch.stack([self.concepts[x] for x in label ],0) for label in label_embed]
        # label_embed = [x.max(0)[0] for x in label_embed]


        self.label_concept_onehot = torch.nn.Parameter(makemultilabels(label_embed, len(self.label_vocabulary)+1).float() )
        self.label_concept_onehot.requires_grad=False

        # self.query_projection = torch.nn.Linear(self.embedding_dim,200)
        # self.key_projection = torch.nn.Linear(self.embedding_dim,200)

        self.input_projection = torch.nn.Linear(self.embedding_dim, self.concept_embedding_dim)

        self.query_projection = torch.nn.Linear(self.embedding_dim, self.att_dim)
        self.key_projection = torch.nn.Linear(self.embedding_dim, self.att_dim)

        self.comparing_space = torch.nn.Linear(self.n_concepts, 256)



        self.build()

    def forward(self, x, return_scores=False):
        with torch.no_grad():
            embeddings = torch.cat(self.embedding(x)[2][(-1 - self.n_layers):-1], -1)
        outputs = self.input_projection(embeddings)

        word_scores = torch.softmax(torch.matmul(self.query_projection(embeddings), self.key_projection(embeddings).permute(0,2,1)), -1).sum(-2)[:,:,None]

        concept_scores = torch.softmax((word_scores*torch.matmul(outputs, self.concepts.float().t())).mean(-2),-1)
        doc = self.comparing_space(concept_scores)
        label = self.comparing_space(self.label_concept_onehot)

        classes = torch.matmul(doc, label.t())

        if return_scores:
            return classes, concept_scores, word_scores
        return classes

