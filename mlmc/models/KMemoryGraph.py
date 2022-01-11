from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from mlmc.models.abstracts.abstract_sentence import SentenceTextClassificationAbstract
import torch
from ..modules.dropout import VerticalDropout
from ..graph import get as gget
from ..modules.module_tfidf import TFIDFAggregation

class NormedLinear(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(NormedLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn((input_dim, output_dim)))
        self.use_bias = bias
        if bias:
            self.bias = torch.nn.Parameter(torch.randn((1, output_dim,)))

        self.g = torch.nn.Parameter(torch.tensor([0.001]))
    def forward(self, x):
        r =  torch.mm(x, self.weight/self.weight.norm(p=2, dim=0, keepdim=True))
        if self.use_bias:
            r = r + self.bias
        return r * self.g

class KMemoryGraph(SentenceTextClassificationAbstract, TextClassificationAbstractZeroShot):
    def __init__(self, similarity="cosine", dropout=0.5,  *args, **kwargs):
        super(KMemoryGraph, self).__init__(*args, **kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.parameter = torch.nn.Linear(self.embeddings_dim,256)
        self.entailment_projection = torch.nn.Linear(3 * self.embeddings_dim, self.embeddings_dim)
        self.entailment_projection2 = torch.nn.Linear(self.embeddings_dim, 1)
        self.project = NormedLinear(self.embeddings_dim, len(self.classes), bias=False)

        self._config["dropout"] = dropout
        self.create_labels(self.classes)
        self._config["similarity"] = similarity

        graph = gget("wordnet")
        self.agg = TFIDFAggregation()


        self.map = {"Sports": ["sport"], "Business":["business"], "World": ["world"], "Sci/Tech": ["science", "technology"] ,
                    "Company":["company"], "EducationalInstitution": ["Education", "institution"], "Artist":["artist"],
                    "Athlete":["athlete"], "OfficeHolder":["officeholder"], "MeanOfTransportation": ["Transportation", "vehicle"],
                    "Building":["building"], "NaturalPlace":["nature", "region", "location"], "Village":["village"],
                    "Animal":["animal"], "Plant":["plant"], "Album":["album"], "Film":["film"], "WrittenWork":["writing", "literature", "work"],
                    "ABBR": ["abbreviation"], "DESC": ["description"], "ENTY":["entity", "person"], "HUM":["human", "person"], "LOC":["location"], "NUM": ["number"],
                    "Society & Culture":["society", "culture"], "Science & Mathematics":["science", "mathematics"], "Health":["health"],
                    "Education & Reference":["Education", "reference"], "Computers & Internet":["computer", "internet"], "Business & Finance": ["business", "finance"],
                    "Entertainment & Music":["entertainment", "music"], "Family & Relationships": ["family", "relationship"], "Politics & Government":["politics", "government"]}

        self.memory = {
            k: [k] +[x for x in sum([list(graph.neighbors(x)) for x in self.map[k]], [])]
            for k in self.classes.keys()
        }

        self.memory_dicts = {}
        self.update_memory()
        self.vdropout = VerticalDropout(0.5)

        self.build()

    def fit(self, train, valid,*args, **kwargs):
        # for x, y in zip(train.x, train.y):
        #     for l in y:
        #         self.memory[l] = list(set(self.memory.get(l, []) + [x]))
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
        # if self.training:
        #     input_embedding = input_embedding + 0.01*torch.rand_like(input_embedding)[:,0,None,0,None].round()*torch.rand_like(input_embedding) #
        #     input_embedding = input_embedding * ((torch.rand_like(input_embedding[:,:,0])>0.05).float()*2 -1)[...,None]

        memory_embedding = {x: self._mean_pooling(memory_embedding[x], self.memory_dicts[x]["attention_mask"]) if memory_embedding[x] is not None else None for x in memory_embedding}

        words, ke, tfidf= self.agg(input_embedding, memory_embedding.values(), x_mask = x["attention_mask"])

        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])
        r2 = torch.stack([self._sim(input_embedding, x).max(-1)[0] for i,(k, x) in enumerate(memory_embedding.items())],-1)
        r = self._sim(ke,label_embedding).squeeze()
        r3 = self._sim(input_embedding,label_embedding).squeeze()
        p = self.project(input_embedding)
        return torch.stack([r,r2,r3,p],-1).mean(-1)#tfidf.max(1)[0].log_softmax(-1)