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

    def forward(self, x):
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

        r1 = self._sim(ke,label_embedding).squeeze()
        r2 = torch.stack([self._sim(input_embedding, x).max(-1)[0] for i,(k, x) in enumerate(memory_embedding.items())],-1)
        r3 = self._sim(input_embedding,label_embedding).squeeze()
        r4 = torch.mm(self._sim(input_embedding,nodes_embedding).squeeze(), (self.adjencies/self.adjencies.norm(1,dim=-1, keepdim=True)).t())
        # p = self.project(input_embedding)
        l = [r1, r2,r3, r4]
        # l = [r2, r4]
        return torch.stack(l,-1).mean(-1)#tfidf.max(1)[0].log_softmax(-1)