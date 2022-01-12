import networkx as nx

from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from mlmc.models.abstracts.abstract_sentence import SentenceTextClassificationAbstract
import torch
from ..modules.dropout import VerticalDropout
from ..graph import get as gget
from ..modules.module_tfidf import TFIDFAggregation
import networkx as nx

class NormedLinear(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(NormedLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn((input_dim, output_dim)))
        self.use_bias = bias
        if bias:
            self.bias = torch.nn.Parameter(torch.randn((1, output_dim,)))

        self.g = torch.nn.Parameter(torch.tensor([0.005]))
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

        # self.entailment_projection = torch.nn.Linear(3 * self.embeddings_dim, self.embeddings_dim)
        # self.entailment_projection2 = NormedLinear(self.embeddings_dim, 1, bias=True)

        self.project = NormedLinear(self.embeddings_dim, len(self.classes), bias=False)

        self._config["dropout"] = dropout
        self._config["similarity"] = similarity

        self.agg = TFIDFAggregation()

        self._config["pos"] = ["a", "s", "n", "v"]
        self._config["depth"] = 2
        self._config["graph"] = "wordnet"
        self.map = {"Sports": ["sport"], "Business":["business"], "World": ["world"], "Sci/Tech": ["science", "technology"] ,
                    "Company":["company"], "EducationalInstitution": ["Education", "institution"], "Artist":["artist"],
                    "Athlete":["athlete"], "OfficeHolder":["officeholder"], "MeanOfTransportation": ["Transportation", "vehicle"],
                    "Building":["building"], "NaturalPlace":["nature", "region", "location"], "Village":["village"],
                    "Animal":["animal"], "Plant":["plant"], "Album":["album"], "Film":["film"], "WrittenWork":["writing", "literature", "work"],
                    "ABBR": ["abbreviation"], "DESC": ["description"], "ENTY":["entity", "person"], "HUM":["human", "person"], "LOC":["location"], "NUM": ["number"],
                    "Society & Culture":["society", "culture"], "Science & Mathematics":["science", "mathematics"], "Health":["health"],
                    "Education & Reference":["Education", "reference"], "Computers & Internet":["computer", "internet"], "Business & Finance": ["business", "finance"],
                    "Entertainment & Music":["entertainment", "music"], "Family & Relationships": ["family", "relationship"], "Politics & Government":["politics", "government"],
                    # "1":["1", "worst", "terrible"], "2":["2","poor", "odd", "simple"], "3":["3", "neutral","ok", "fine"], "4":["4", "bold", "worth", "good", "nice"], "5":["5","amazing", "excellent", "wow"],
                    "1":["1"], "2":["2"], "3":["3"], "4":["4",], "5":["5"],
                    "ENTY:sport": ["entity", "sport"], "ENTY:dismed": ["disease", "medicine"], "LOC:city": ["location", "city"],"DESC:reason": ["description","reason"],
                    "NUM:other": ["number"],"LOC:state": ["location", "state"],"NUM:speed": ["number", "speed"],"NUM:ord": ["number", "order", "rank"],
                    "ENTY:event": ["event"],"ENTY:substance": ["element", "substance"],"NUM:perc": ["number", "percentage", "fraction"],
                    "ENTY:product": ["product"],"ENTY:animal": ["animal"],"DESC:manner": ["description", "manner", "action"],
                    "ENTY:cremat": ["creative","invention","book"],"ENTY:color": ["color"],"ENTY:techmeth": ["technique", "method"],
                    "NUM:dist": ["number",  "distance", "measure"],"NUM:weight": ["number", "weight"],"LOC:mount": ["location", "mountain"],
                    "HUM:title": ["person", "title"],"HUM:gr": ["group", "organization", "person"],
                    "HUM:desc": ["person", "description"],"ABBR:abb": ["abbreviation"],
                    "ENTY:currency": ["currency"],"DESC:def": ["description", "definition"],"NUM:code": ["number", "code"],"LOC:other": ["location"],
                    "ENTY:other": ["entity", "other"],"ENTY:body": ["body", "organ"],"ENTY:instru": ["music", "instrument"],
                    "ENTY:termeq": ["synonym"],"NUM:money": ["number", "money", "price"],"NUM:temp": ["number", "temperature"],
                    "LOC:country": ["location", "country"],"ABBR:exp": ["abbreviation", "expression"],"ENTY:symbol": ["symbol", "sign"],
                    "ENTY:religion":["entity" ,"religion"],"HUM:ind": ["individual", "person"],"ENTY:letter": ["letter", "character"],
                    "NUM:date": ["number", "date"],"ENTY:lang": ["language"],"ENTY:veh": ["vehicle"],
                    "NUM:count": ["number", "count"],"ENTY:word": ["word", "special", "property"],"NUM:period": ["number", "time period", "time"],
                    "ENTY:plant": ["entity", "plant"],"ENTY:food": ["entity", "food"],"NUM:volsize": ["number", "volume", "size"],
                    "DESC:desc": ["description"],
                    }
        self.create_labels(self.classes)
        self.vdropout = VerticalDropout(0.5)
        self._classifier_weight = torch.nn.Parameter(torch.tensor([0.01]))
        self.build()

    def fit(self, train, valid,*args, **kwargs):
        # for x, y in zip(train.x, train.y):3
        #     for l in y:
        #         self.memory[l] = list(set(self.memory.get(l, []) + [x]))
        # self.update_memory()

        return super().fit(train, valid, *args, **kwargs)


    def update_memory(self):
        """
        Method to change the current target variables
        Args:
            classes: Dictionary of class mapping like {"label1": 0, "label2":1, ...}

        Returns:

        """
        graph = gget(self._config["graph"])

        self.memory = {
            k: [k] +self.map[k]+[x for x in sum([list(graph.neighbors(x)) for x in self.map[k]] , [])] # if graph.nodes(True)[x]["pos"] in self._config["pos"]
            for k in self.classes.keys()
        }

        subgraph = {
            k: graph.subgraph(self.map[k] + [x for x in sum([list(graph.neighbors(x)) for x in self.map[k] ], [])])
            for k in self.classes.keys()
        }

        self.g = nx.OrderedDiGraph()
        for k, v in subgraph.items():
            self.g = nx.compose(self.g,v)
            self.g.add_node(k)
            self.g.add_edges_from([(n,k) for n in v.nodes])
            self.g.add_edge(k,k)
            self.g.add_edges_from([(k,n) for n in v.nodes])

        self.memory_dicts = {}

        self.memory_dicts = {k:self.label_embed(ex) for k, ex in self.memory.items() }
        self._node_list = sorted(list(self.g.nodes))
        self.nodes = self.transform(self._node_list)
        self._class_nodes = {k:self._node_list.index(k) for k in self.classes.keys()}
        adj = nx.adj_matrix(self.g, self._node_list)
        self.adjencies = torch.nn.Parameter(torch.cat([torch.tensor(adj[i].toarray()) for i in self._class_nodes.values()],0).float()).to(self.device)
        self.adjencies = self.adjencies.detach()

    def create_labels(self, classes: dict):
        super().create_labels(classes)
        self.update_memory()

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

    def _entailment(self, x, y,):
        b = tuple([1]*(len(x.shape)-2))
        e = self.entailment_projection(self.dropout(torch.cat([
            x.unsqueeze(-2).repeat(*(b+ (1, y.shape[0], 1))),
            y.unsqueeze(-3).repeat(*(b+ (x.shape[0], 1, 1))),
            (x.unsqueeze(-2) - y.unsqueeze(-3)).abs()
        ], -1)))
        r = self.entailment_projection2(e).squeeze(-1)
        if self._config["target"] == "entailment":
            r = r.diag()
        return r

    def forward(self, x,return_keywords=False):
        input_embedding = self.vdropout(self.embedding(**x)[0])
        label_embedding = self.dropout(self.embedding(**self.label_dict)[0])
        nodes_embedding = self.embedding(**self.nodes)[0]
        memory_embedding = {x:self.embedding(**self.memory_dicts.get(x))[0] if x in self.memory_dicts else None for x in self.classes.keys()}


        if self.training:
            input_embedding = input_embedding + 0.01*torch.rand_like(input_embedding)[:,0,None,0,None].round()*torch.rand_like(input_embedding) #
            input_embedding = input_embedding * ((torch.rand_like(input_embedding[:,:,0])>0.05).float()*2 -1)[...,None]


        memory_embedding = {x: self._mean_pooling(memory_embedding[x], self.memory_dicts[x]["attention_mask"]) if memory_embedding[x] is not None else None for x in memory_embedding}

        words, ke, tfidf= self.agg(input_embedding, memory_embedding.values(), x_mask = x["attention_mask"])

        nodes_embedding = self._mean_pooling(nodes_embedding, self.nodes["attention_mask"])
        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])


        r1 = self._sim(ke,label_embedding).squeeze() # weighted-similarity
        r2 = torch.stack([self._sim(input_embedding, x).max(-1)[0] for i,(k, x) in enumerate(memory_embedding.items())],-1) # keyword-similarity-max
        r3 = self._sim(input_embedding,label_embedding).squeeze() # pooled-similarity
        r4 = torch.mm(self._sim(input_embedding,nodes_embedding).squeeze(), (self.adjencies/self.adjencies.norm(1,dim=-1, keepdim=True)).t()) # keyword-similarity-mean
        p = self._classifier_weight * self._entailment(input_embedding, label_embedding) # classifier
        l = [r1, r2,r3, r4, p]
        # l = [r3,r2, r4]
        scores = torch.stack(l,-1).mean(-1)
        if return_keywords:
            return scores, tfidf.softmax(-1)
        return scores#tfidf.max(1)[0].log_softmax(-1)

    def transform(self, x, max_length=None, return_tokens=False) -> dict:
        if max_length is None:
            max_length = self._config["max_len"]
        r = {k: v.to(self.device) for k, v in
                self.tokenizer(x, padding=True, max_length=max_length, truncation=True,
                               add_special_tokens=True, return_tensors='pt').items()}
        if return_tokens:
            return r, [self.tokenizer.tokenize(s) for s in x]
        return r

    def keywords(self, x, y, n=10):
        self.eval()
        with torch.no_grad():
            i, tokens = self.transform(x, return_tokens=True)
            scores, keywords = self.forward(i, return_keywords=True)
        import matplotlib.pyplot as plt
        sorted_scores, prediction  = scores.sort(-1)
        idx = scores.argmax(-1)

        label_specific_scores = torch.stack([k[i] for k, i in zip(keywords, idx)])
        keywords = [list(zip(t,l[1:(1+len(t))].tolist())) for t,l in zip(tokens, label_specific_scores)]
        keywords_new = []
        for l in keywords:
            new_list = []
            new_tuple = [[], 0]
            for i in range(1, 1+len(l)):
                new_tuple[0].append(l[-i][0])
                new_tuple[1] += l[-i][1]

                if not l[-i][0].startswith("##"):
                    new_tuple[0] = "".join(new_tuple[0][::-1]).replace("##", "")
                    new_list.append(tuple(new_tuple))
                    new_tuple = [[], 0]
            keywords_new.append(new_list)

        import numpy as np
        prediction = [[self.classes_rev[x] for x in y] for y in prediction.detach().cpu().tolist()]
        binary = np.array([[p in c for p in pred] for c, pred in zip(y, prediction)])

        import seaborn as sns
        ax = sns.heatmap((sorted_scores.softmax(-1).cpu()+binary), annot=np.array(prediction), fmt="")
        plt.show()
        return [(p[-1], t, sorted(x, key=lambda x: -x[1])[:n]) for p, t,x in zip(prediction, y, keywords_new)]


    def scores(self, x):
        """
        Returns 2D tensor with length of x and number of labels as shape: (N, L)
        Args:
            x:

        Returns:

        """
        self.eval()
        assert not (self._config["target"] == "single" and self._config["threshold"] != "max"), \
            "You are running single target mode and predicting not in max mode."

        if not hasattr(self, "classes_rev"):
            self.classes_rev = {v: k for k, v in self.classes.items()}
        x = self.transform(x)
        with torch.no_grad():
            output = self.act(self(x))
            # output = 0.5*(output+1)
        self.train()
        return output