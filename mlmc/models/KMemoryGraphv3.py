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
    def __init__(self, similarity="cosine", dropout=0.5,  graph="wordnet", *args, **kwargs):
        super(KMemoryGraph, self).__init__(*args, **kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.parameter = torch.nn.Linear(self.embeddings_dim,256)
        self.entailment_projection = torch.nn.Linear(3 * self.embeddings_dim, self.embeddings_dim)
        self.entailment_projection2 = torch.nn.Linear(self.embeddings_dim, 1)

        self.project = NormedLinear(self.embeddings_dim, len(self.classes), bias=False)

        self._config["dropout"] = dropout
        self._config["similarity"] = similarity

        self.agg = TFIDFAggregation()

        self._config["scoring"] = [ "keyword_similarity_max", "pooled_similarity", "keyword_similiarity_mean", "fallback_classifier", "weighted_similarity"]
        self._config["pos"] = ["a", "s", "n", "v"]
        self._config["depth"] = 2
        self._config["graph"] = graph
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
                    "ENTY:sport": ["entity", "sport"], "ENTY:dismed": ["entity","disease", "medicine"], "LOC:city": ["location", "city"],
                    "DESC:reason": ["description","reason"],
                    "NUM:other": ["number"],"LOC:state": ["location", "state"],"NUM:speed": ["number", "speed"],"NUM:ord": ["number", "order", "rank"],
                    "ENTY:event": ["entity","event"],"ENTY:substance": ["entity","element", "substance"],"NUM:perc": ["number", "percentage", "fraction"],
                    "ENTY:product": ["entity","product"],"ENTY:animal": ["entity","animal"],"DESC:manner": ["description", "manner", "action"],
                    "ENTY:cremat": ["entity","creative","invention","book"],"ENTY:color": ["entity","color"],"ENTY:techmeth": ["entity","technique", "method"],
                    "NUM:dist": ["number",  "distance", "measure"],"NUM:weight": ["number", "weight"],"LOC:mount": ["location", "mountain"],
                    "HUM:title": ["person", "title"],"HUM:gr": ["group", "organization", "person"],
                    "HUM:desc": ["person", "description"],"ABBR:abb": ["abbreviation"],
                    "ENTY:currency": ["entity","currency"],"DESC:def": ["description", "definition"],"NUM:code": ["number", "code"],"LOC:other": ["location"],
                    "ENTY:other": ["entity", "other"],"ENTY:body": ["entity","body", "organ"],"ENTY:instru": ["entity","music", "instrument"],
                    "ENTY:termeq": ["entity","synonym"],"NUM:money": ["number", "money", "price"],"NUM:temp": ["number", "temperature"],
                    "LOC:country": ["location", "country"],"ABBR:exp": ["abbreviation", "expression"],"ENTY:symbol": ["entity","symbol", "sign"],
                    "ENTY:religion":["entity" ,"religion"],"HUM:ind": ["individual", "person"],"ENTY:letter": ["entity","letter", "character"],
                    "NUM:date": ["number", "date"],"ENTY:lang": ["entity","language"],"ENTY:veh": ["entity","vehicle"],
                    "NUM:count": ["number", "count"],"ENTY:word": ["entity","word", "special", "property"],"NUM:period": ["number", "time period", "time"],
                    "ENTY:plant": ["entity","plant"],"ENTY:food": ["entity", "food"],"NUM:volsize": ["number", "volume", "size"],
                    "DESC:desc": ["description"],
                    }
        import re
        self.map = {'Teen & Young Adult Mystery & Suspense': ['teen', 'young', 'adult', 'mystery', 'suspense'],
                    'Regency Romance': ['regency', 'romance'], 'Baking & Desserts': ['baking', 'dessert'],
                    'Teen & Young Adult Fantasy Fiction': ['teen', 'young', 'adult', 'fantasy', 'fiction'],
                    'Travel: Australia & Oceania': ['travel', 'Australia', 'Oceania'],
                    'Children’s Chapter Books': ['child', 'chapter', 'book'],
                    'Science Fiction': ['science', 'fiction'], 'Travel: Europe': ['travel', 'Europe'],
                    'North American World History': ['North American', 'world', 'history'],
                    'Graphic Novels & Manga': ['graphic', 'novel', 'anime'],
                    'Travel: Central & South America': ['travel', 'central', 'South America'],
                    'Performing Arts': ['performing arts'], '20th Century U.S. History': ['20th', 'century', 'U.S.', 'history'],
                    'Arts & Entertainment Biographies & Memoirs': ['arts', 'entertainment', 'biography', 'memoir'],
                    'Travel: Caribbean & Mexico': ['travel', 'Caribbean', 'Mexico'], 'Photography': ['photography'],
                    'Alternative Therapies': ['alternative', 'therapy'], 'Nonfiction Classics': ['nonfiction', 'classics'],
                    'Paranormal Fiction': ['paranormal', 'fiction'], 'Military Fiction': ['military', 'fiction'],
                    'Colonial/Revolutionary Period': ['colonial', 'revolutionary', 'period'], 'Politics': ['politics'],
                    'Test Preparation': ['test', 'preparation'],
                    'Teen & Young Adult Historical Fiction': ['teen', 'young', 'adult', 'historical', 'fiction'],
                    'U.S. History': ['U.S.', 'history'], 'Children’s Picture Books': ['child', 'picture', 'book'],
                    'Fiction Classics': ['fiction', 'classics'], 'Ancient World History': ['ancient', 'world', 'history'],
                    'Classics': ['classics'], 'Business': ['business'], 'Military Science Fiction': ['military', 'science fiction'],
                    'World War I Military History': ['world war', 'i', 'military', 'history'],
                    'Fiction': ['fiction'], 'Paranormal Romance': ['paranormal', 'romance'], 'Women’s Fiction': ['woman', 'fiction'],
                    'Crime Mysteries': ['crime', 'mystery'], 'Design': ['design'], 'Personal Growth': ['personal', 'growth'],
                    'Marketing': ['marketing'], 'Travel': ['travel'], 'Personal Finance': ['personal', 'finance'], 'Crafts & Hobbies': ['craft', 'hobby'],
                    'Religion & Philosophy': ['religion', 'philosophy'], 'Cooking Methods': ['cooking', 'method'],
                    'Fairy Tales': ['fairy tale'], 'Travel Writing': ['travel', 'writing'],
                    'Nonfiction': ['nonfiction'], 'Science': ['science'], 'World History': ['world', 'history'],
                    'Fantasy': ['fantasy'], 'Language': ['language'], 'Technology': ['technology'],
                    'Western Romance': ['western', 'romance'], 'Biography & Memoir': ['biography', 'memoir'],
                    'Home & Garden': ['home', 'garden'], 'Contemporary Fantasy': ['contemporary', 'fantasy'],
                    'Spiritual Fiction': ['spiritual', 'fiction'], 'Suspense Romance': ['suspense', 'romance'],
                    'Art': ['art'], 'Literary Figure Biographies & Memoirs': ['literary', 'figure', 'biography', 'memoir'],
                    'Mystery & Suspense': ['mystery', 'suspense'], 'Civil War Period': ['civil war', 'period'],
                    'Teen & Young Adult Fiction': ['teen', 'young', 'adult', 'fiction'],
                    'Historical Figure Biographies & Memoirs': ['historical', 'figure', 'biography', 'memoir'],
                    'African World History': ['African', 'world', 'history'], 'Health & Reference': ['health', 'reference'],
                    'Childrens Media Tie-In Books': ['child', 'audio', 'video', 'tie-in', 'book'],
                    'Regional & Ethnic Cooking': ['regional', 'ethnic', 'cooking'], 'Romance': ['romance'],
                    'Children’s Boxed Sets': ['child', 'boxed', 'set'], 'Teen & Young Adult Social Issues': ['teen', 'young', 'adult', 'social', 'issue'],
                    'Economics': ['economics'], 'Wine & Beverage': ['wine', 'beverage'],
                    'Latin American World History': ['Latin American', 'world', 'history'], 'Pets': ['pet'], 'Music': ['music'],
                    'Urban Fantasy': ['urban', 'fantasy'], 'Travel: Africa': ['travel', 'Africa'],
                    'Health & Fitness': ['health', 'fitness'], 'Noir Mysteries': ['film noir', 'mystery'],
                    'Children’s Middle Grade Historical Books': ['child', 'middle', 'grade', 'historical', 'book'],
                    'Children’s Middle Grade Fantasy & Magical Books': ['child', 'middle', 'grade', 'fantasy', 'magical', 'book'],
                    'Children’s Books': ['child', 'book'], 'Teen & Young Adult Action & Adventure': ['teen', 'young', 'adult', 'action', 'adventure'],
                    'Historical Romance': ['historical', 'romance'], 'Teen & Young Adult': ['teen', 'young', 'adult'],
                    'Children’s Middle Grade Mystery & Detective Books': ['child', 'middle', 'grade', 'mystery', 'detective', 'book'],
                    'Epic Fantasy': ['epic', 'fantasy'], 'Humor': ['humor'], 'Self-Improvement': ['self-improvement'],
                    'Historical Fiction': ['historical', 'fiction'], 'Games': ['game'], 'Literary Criticism': ['literary', 'criticism', 'literary criticism'],
                    'Asian World History': ['Asian', 'world', 'history'], 'Middle Eastern World History': ['Middle Eastern', 'world', 'history'],
                    'Teen & Young Adult Romance': ['teen', 'young', 'adult', 'romance'], 'Writing': ['writing'],
                    'Travel: USA & Canada': ['travel', 'USA', 'Canada'], 'Literary Fiction': ['literary', 'fiction'], 'Popular Science': ['popular', 'science'],
                    'Literary Collections': ['literary', 'collection'], 'Children’s Middle Grade Books': ['child', 'middle', 'grade', 'book'],
                    'Erotica': ['erotica'], 'Crafts, Home & Garden': ['craft', 'home', 'garden'],
                    'Children’s Middle Grade Action & Adventure Books': ['child', 'middle', 'grade', 'action', 'adventure', 'book'],
                    'Travel: Asia': ['travel', 'Asia'], 'Cooking': ['cooking'], 'Cozy Mysteries': ['cozy', 'mystery'], 'History': ['history'],
                    'Espionage Mysteries': ['espionage', 'mystery'], 'World War II Military History': ['world war', 'ii', 'military', 'history'],
                    'Suspense & Thriller': ['suspense', 'thriller'], '19th Century U.S. History': ['19th', 'century', 'U.S.', 'history'],
                    'Space Opera': ['space', 'opera'], 'Diet & Nutrition': ['diet', 'nutrition'], 'Religion': ['religion'],
                    'Arts & Entertainment': ['art', 'entertainment'], 'Specialty Travel': ['specialty', 'travel'],
                    'Weddings': ['wedding'], 'Teen & Young Adult Nonfiction': ['teen', 'young', 'adult', 'nonfiction'],
                    '21st Century U.S. History': ['21st', 'century', 'U.S.', 'history'], 'Gothic & Horror': ['gothic', 'horror'],
                    'Domestic Politics': ['domestic', 'politics'], 'Reference': ['reference'], 'Beauty': ['beauty'],
                    'Sports': ['sport'], 'Western Fiction': ['Western', 'fiction'],
                    'Teen & Young Adult Science Fiction': ['teen', 'young', 'adult', 'science fiction'],
                    'Philosophy': ['philosophy'], 'Parenting': ['parent', 'raise'],
                    'Native American History': ['Native American', 'history'], 'Poetry': ['poetry'], 'Psychology': ['psychology'],
                    'Inspiration & Motivation': ['inspiration', 'motivation'], 'Step Into Reading': ['beginner', "start", 'reading'],
                    'Exercise': ['exercise'], 'Bibles': ['bible'], 'Travel: Middle East': ['travel', 'Middle East'],
                    'New Adult Romance': ['new', 'adult', 'romance'], 'Contemporary Romance': ['contemporary', 'romance'],
                    'Military History': ['military', 'history'], 'Cyber Punk': ["cyberpunk"], 'Film': ['film'],
                    'Children’s Middle Grade Sports Books': ['child', 'middle', 'grade', 'sport', 'book'],
                    'European World History': ['European', 'world', 'history'],
                    'Political Figure Biographies & Memoirs': ['political', 'figure', 'biography', 'memoir'],
                    'Children’s Activity & Novelty Books': ['child', 'activity', 'novelty', 'book'],
                    '1950 – Present Military History': ['present', 'military', 'history'],
                    'Children’s Board Books': ['child', 'board', 'book'], 'World Politics': ['world', 'politics'],
                    'Food Memoir & Travel': ['food', 'memoir', 'travel'], 'Management': ['management']}

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

        weighted_similarity = self._sim(ke,label_embedding).squeeze() # weighted-similarity

        keyword_similarity_max = torch.stack([self._sim(input_embedding, x).max(-1)[0] for i,(k, x) in enumerate(memory_embedding.items())],-1) # keyword-similarity-max
        pooled_similarity = self._sim(input_embedding,label_embedding).squeeze() # pooled-similarity
        keyword_similiarity_mean = torch.mm(self._sim(input_embedding,nodes_embedding).squeeze(), (self.adjencies/self.adjencies.norm(1,dim=-1, keepdim=True)).t()) # keyword-similarity-mean
        fallback_classifier = self._classifier_weight * self._entailment(input_embedding, label_embedding) # classifier
        l = []
        if "keyword_similarity_max" in self._config["scoring"]:
            l.append(keyword_similarity_max)
        if "pooled_similarity" in self._config["scoring"]:
            l.append(pooled_similarity)
        if "keyword_similiarity_mean" in self._config["scoring"]:
            l.append(keyword_similiarity_mean)
        if "fallback_classifier" in self._config["scoring"]:
            l.append(fallback_classifier)
        if "weighted_similarity" in self._config["scoring"]:
            l.append(weighted_similarity)
        scores = torch.stack(l,-1).mean(-1)
        return scores