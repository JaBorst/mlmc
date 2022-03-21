import networkx as nx

from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from mlmc.models.abstracts.abstract_sentence import SentenceTextClassificationAbstract
import torch
from mlmc.modules.dropout import VerticalDropout
from mlmc.graph import get as gget
from mlmc.modules.module_tfidf import TFIDFAggregation
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
    def __init__(self, similarity="cosine", dropout=0.5, entropy=True,
                 measures=None,depth=1,fallback_classifier = False,
                 graph="wordnet", *args, **kwargs):
        super(KMemoryGraph, self).__init__(*args, **kwargs)
        if measures is None:
            measures= ["pooled_similarity", "keyword_similiarity_mean"]

        if fallback_classifier:
            measures = measures + ["fallback_classifier"]

        self._config["fallback_classifier"] = fallback_classifier
        self.dropout = torch.nn.Dropout(dropout)
        self.parameter = torch.nn.Linear(self.embeddings_dim,256)
        self.entailment_projection = torch.nn.Linear(3 * self.embeddings_dim, self.embeddings_dim)
        self.entailment_projection2 = torch.nn.Linear(self.embeddings_dim, 1)

        self.project = NormedLinear(self.embeddings_dim, len(self.classes), bias=False)

        self._config["dropout"] = dropout
        self._config["similarity"] = similarity
        self._config["entropy"] = entropy

        self.set_scoring(measures)
        self._config["pos"] = ["a", "s", "n", "v"]
        self._config["depth"] = depth
        self._config["graph"] = graph
        from ....graph.helpers import keywordmap
        self.map = keywordmap

        self.create_labels(self.classes)
        self.vdropout = VerticalDropout(0.5)
        self._classifier_weight = torch.nn.Parameter(torch.tensor([0.01]))
        self.build()


    def set_scoring(self, measures):
        self._config["scoring"] = measures

    def _get_graph(self):
        graph = gget(self._config["graph"])

        subgraph = {
            k: graph.subgraph(self.map[k] + [x for x in
                                             sum([list(graph.neighbors(x)) if x in graph else [x] for x in self.map[k]],
                                                 [])])
            for k in self.classes.keys()
        }
        g = nx.OrderedGraph()
        for k, v in subgraph.items():
            g = nx.compose(g, v)
            g.add_node(k)
            g.add_edges_from([(n, k) for n in v.nodes])
            g.add_edge(k, k)

        if self._config["depth"] > 1:
            for _ in range(self._config["depth"] - 1):
                g = nx.compose(g, graph.subgraph(sum([list(graph.neighbors(n)) for n in g if n in graph], [])))
        return g

    def update_memory(self):
        """
        Method to change the current target variables
        Args:
            classes: Dictionary of class mapping like {"label1": 0, "label2":1, ...}

        Returns:

        """
        self.g = self._get_graph()
        self._node_list = sorted(list(self.g.nodes))
        self.nodes = self.transform(self._node_list)
        self._class_nodes = {k:self._node_list.index(k) for k in self.classes.keys()}
        adj = nx.adj_matrix(self.g, self._node_list)

        adj = adj / adj.sum(-1)
        self.adjencies = torch.nn.Parameter(torch.cat([torch.FloatTensor(adj[i]) for i in self._class_nodes.values()],0).float()).to(self.device)
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

    def transform(self, x, max_length=None) -> dict:
        if max_length is None:
            max_length = self._config["max_len"]
        r = {k: v.to(self.device) for k, v in
                self.tokenizer(x, padding=True, max_length=max_length, truncation=True,
                               add_special_tokens=True, return_tensors='pt').items()}
        r["text"] = [self.tokenizer.tokenize(s) for s in x]
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

    def forward(self, x, kw=False):
        input_embedding = self.dropout(self.embedding(**{k:x[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']})[0])
        label_embedding = self.dropout(self.embedding(**{k:self.label_dict[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']})[0])
        nodes_embedding = self.dropout(self.embedding(**{k:self.nodes[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']})[0])


        if self.training:
            input_embedding = input_embedding + 0.01 * torch.rand_like(input_embedding)[:, 0, None, 0,
                                                       None].round() * torch.rand_like(input_embedding)  #
            input_embedding = input_embedding * ((torch.rand_like(input_embedding[:, :, 0]) > 0.05).float() * 2 - 1)[..., None]


        tmp = self._mean_pooling(nodes_embedding, self.nodes["attention_mask"])
        text_kw = torch.einsum("bse,le->bls",
                               input_embedding / input_embedding.norm(p=2, dim=-1, keepdim=True),
                               tmp / tmp.norm(p=2, dim=-1, keepdim=True)).max(1)[0]
        ke = torch.einsum("bse, bs -> be", input_embedding, text_kw.softmax(-1))

        nodes_embedding = self._mean_pooling(nodes_embedding, self.nodes["attention_mask"])
        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])


        l = []
        keyword_similiarity = self._sim(input_embedding, nodes_embedding)
        if "keyword_similarity_max" in self._config["scoring"]:
            keyword_similarity_max = torch.stack([keyword_similiarity[:, torch.where(x != 0)[0]].max(-1)[0] for x in self.adjencies],1)
            l.append(keyword_similarity_max)
        if "pooled_similarity" in self._config["scoring"]:
            pooled_similarity = self._sim(input_embedding, label_embedding).squeeze(-1)  # pooled-similarity
            l.append(pooled_similarity)
        if "keyword_similiarity_mean" in self._config["scoring"]:
            keyword_similiarity_mean = torch.mm(keyword_similiarity, self.adjencies.t())
            l.append(keyword_similiarity_mean)
        if "fallback_classifier" in self._config["scoring"]:
            fallback_classifier = self._classifier_weight * self._entailment(input_embedding,
                                                                             label_embedding)  # classifier
            l.append(fallback_classifier)
        if "weighted_similarity" in self._config["scoring"]:
            weighted_similarity = self._sim(ke, label_embedding).squeeze(-1) # weighted_similarity
            l.append(weighted_similarity)

        scores = torch.stack(l,-1).mean(-1)
        if kw:
            return scores, keyword_similiarity, text_kw
        return scores

    def keywords(self, x, n = 10):
        self.eval()
        with torch.no_grad():
            tok = self.transform(x)
            scores, graphkw, text_kw = self.forward(tok, kw=True)

        gkw = [[list(self._node_list)[i] for i in x] for x in graphkw.softmax(-1).topk(n, dim=-1, )[1]]
        labels = [[self.classes_rev[x.item()]] for x in scores.argmax(-1).cpu()]
        keywords = [list(zip(t, l[1:(1 + len(t))].tolist())) for t, l in zip(tok["text"], text_kw)]
        keywords_new = []
        for l in keywords:
            new_list = []
            new_tuple = [[], 0]
            for i in range(1, 1 + len(l)):
                new_tuple[0].append(l[-i][0])
                new_tuple[1] += l[-i][1]

                if not l[-i][0].startswith("##"):
                    new_tuple[0] = "".join(new_tuple[0][::-1]).replace("##", "")
                    new_list.append(tuple(new_tuple))
                    new_tuple = [[], 0]
            keywords_new.append(new_list)
        tkw = [sorted(x, key=lambda x: -x[1])[:n] for x in keywords_new]
        return labels, gkw, tkw