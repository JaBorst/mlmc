from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from mlmc.models.abstracts.abstract_sentence import SentenceTextClassificationAbstract
import torch
from ..modules.dropout import VerticalDropout


class KMemory(SentenceTextClassificationAbstract, TextClassificationAbstractZeroShot):
    def __init__(self, similarity="cosine", dropout=0.5,  *args, **kwargs):
        super(KMemory, self).__init__(*args, **kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.parameter = torch.nn.Linear(self.embeddings_dim,256)
        self.entailment_projection = torch.nn.Linear(3 * self.embeddings_dim, self.embeddings_dim)
        self.entailment_projection2 = torch.nn.Linear(self.embeddings_dim, 1)
        self._config["dropout"] = dropout
        self.create_labels(self.classes)
        self._config["similarity"] = similarity
        self.memory = {k:[self._config["sformatter"](v)] for k,v in self.classes.items()}
        self.memory_dicts = {}
        self.update_memory()
        self.vdropout = VerticalDropout(0.5)

        self.build()

    def fit(self, train, valid,*args, **kwargs):
        for x, y in zip(train.x, train.y):
            for l in y:
                self.memory[l] = list(set(self.memory.get(l, []) + [x]))
        self.update_memory()

        return super().fit(train, valid, *args, **kwargs)


    def update_memory(self):
        """
        Method to change the current target variables
        Args:
            classes: Dictionary of class mapping like {"label1": 0, "label2":1, ...}

        Returns:

        """
        if len(self.memory) != 0: # To ensure we can initialize the model without specifying classes
            self.memory_dicts = {k:self.label_embed(ex) for k, ex in self.memory.items() }



    def _sim(self, x, y):
        if self._config["similarity"] == "cosine":
            x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            y = y / (y.norm(p=2, dim=-1, keepdim=True) + 1e-25)
            r = (x[:, None] * y[None]).sum(-1)
            r = torch.log(0.5 * (r + 1))
        elif self._config["similarity"] == "scalar":
            r = (x[:, None] * y[None]).sum(-1)
        elif self._config["similarity"] == "manhattan":
            r = - (x[:, None] * y[None]).abs().sum(-1)
        elif self._config["similarity"] == "entailment":
            e = self.entailment_projection(self.dropout(torch.cat([
                x[:, None].repeat(1, y.shape[0], 1),
                y[None].repeat(x.shape[0], 1, 1),
                (x[:, None] - y[None]).abs()
            ], -1)))
            r = self.entailment_projection2(e).squeeze(-1)
            if self._config["target"] == "entailment":
                r = r.diag()
        return r

    def forward(self, x):
        input_embedding = self.vdropout(self.embedding(**x)[0])
        label_embedding = self.dropout(self.embedding(**self.label_dict)[0])
        memory_embedding = {x:self.embedding(**self.memory_dicts.get(x))[0] if x in self.memory_dicts else None for x in self.classes.keys()}

        if self.training:
            input_embedding = input_embedding + 0.1*torch.rand_like(input_embedding)[:,0,None,0,None]*torch.rand_like(input_embedding) #
            input_embedding = input_embedding * ((torch.rand_like(input_embedding[:,:,0])>0.05).float()*2 -1)[...,None]

        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])
        memory_embedding = {x: self._mean_pooling(memory_embedding[x], self.memory_dicts[x]["attention_mask"]) if memory_embedding[x] is not None else None for x in memory_embedding}

        r = self._sim(input_embedding, label_embedding)
        r2 = torch.stack([self._sim(input_embedding, x).max(-1)[0] if x is not None else (r[:,i]) for i,(k, x) in enumerate(memory_embedding.items())],-1)
        # return 0.5*(r + r2)
        return r2