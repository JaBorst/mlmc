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
        self.agg = TFIDFAggregation()

        self.set_scoring(measures)
        self._config["depth"] = depth
        self._config["graph"] = graph
        from ....graph.helpers import keywordmap

        self.map = keywordmap# {"Sports": ["sport"], "Business": ["business", "financial", "tech"], "World": ["world", "war", "countries", "financial"], "Sci/Tech": ["science", "technology", "software", "computer"]}

        self.create_labels(self.classes)
        self.vdropout = VerticalDropout(0.5)
        self._classifier_weight = torch.nn.Parameter(torch.tensor([0.01]))
        self.build()


    def set_scoring(self, measures):
        self._config["scoring"] = measures

    def _get_graph(self):
        graph = gget(self._config["graph"])#.to_undirected()

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

        remove = [node for node, degree in dict(g.degree()).items() if degree < 2]
        g.remove_nodes_from(remove)


        return g

    def update_memory(self):
        """
        Method to change the current target variables
        Args:
            classes: Dictionary of class mapping like {"label1": 0, "label2":1, ...}

        Returns:

        """
        self.g = self._get_graph()

        self.build()

        self._node_list = sorted(list(self.g.nodes))
        self.nodes = self.transform([x.replace("_", " ") for x in self._node_list])


        self._class_nodes = {k:self._node_list.index(k) for k in self.classes.keys()}
        adj = nx.adj_matrix(self.g, self._node_list)

        self.adjacency_shape = adj.shape
        with torch.no_grad():
            nodes_embedding = self.embedding(**{k: self.nodes[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']})[0].mean(1)
            nodes_embedding = nodes_embedding/nodes_embedding.norm(2, dim=-1, keepdim=True)
            s = torch.mm(nodes_embedding, nodes_embedding.t()) * torch.FloatTensor(adj.toarray()).to(self.device)
            s = s / s.sum(-1)
            adj = s.to_sparse()

        self.adj = torch.nn.Parameter(adj).to(self.device)
        self.adjencies = torch.nn.Parameter(torch.stack([adj[i] for i in self._class_nodes.values()],0).float().to_dense()).to(self.device).detach()
        # self.adjencies = self.adjencies

        l = {}
        for cls in self.classes.keys():
            l[cls] = []
            for n in self._node_list:
                try:
                    l[cls].append(len(nx.shortest_path(self.g, cls, n)))
                except nx.exception.NetworkXNoPath:
                    l[cls].append(0)

        d = 1/(torch.tensor(list(l.values()))-1)
        d = d.clamp(0,1)
        self.distance = torch.nn.Parameter(d).to(self.device)




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
        input_embedding_t = self.dropout(self.embedding(**{k:x[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']})[0])
        label_embedding = self.dropout(self.embedding(**{k:self.label_dict[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']})[0])
        nodes_embedding = self.dropout(self.embedding(**{k:self.nodes[k] for k in ['input_ids', 'token_type_ids', 'attention_mask']})[0])

        # input_embedding_t = self.dropout(self.embedding(**x)[0])
        # label_embedding = self.dropout(self.embedding(**self.label_dict)[0])
        # nodes_embedding = self.dropout(self.embedding(**self.nodes)[0])

        input_embedding=input_embedding_t
        if self.training:
            input_embedding = input_embedding + 0.1 * torch.rand_like(input_embedding)[:, 0, None, 0,
                                                       None].round() * torch.rand_like(input_embedding)  #
            input_embedding = input_embedding * ((torch.rand_like(input_embedding[:, :, 0]) > 0.05).float() * 2 - 1)[..., None]


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

        if "distribution_cmp" in self._config["scoring"]:
            nodes_embedding_norm = nodes_embedding / nodes_embedding.norm(2, dim = -1, keepdim=True)
            input_embedding = input_embedding / input_embedding.norm(2, dim = -1, keepdim=True)
            label_embedding = label_embedding / label_embedding.norm(2, dim = -1, keepdim=True)

            in_distr = torch.mm(input_embedding, nodes_embedding_norm.t())
            l_distr = torch.mm(label_embedding, nodes_embedding_norm.t()) #
            # l.append((in_distr[:,None] * self.distance[None]).sum(-1).log_softmax(-1))
            with torch.no_grad():
                mean = in_distr.mean(-1, keepdim=True)
                std = in_distr.std(-1, keepdim=True)
                mask = (in_distr < mean - 2 * std) | (in_distr > mean + 2 * std)  #


            in_distr = in_distr * mask



            l.append(self._sim(in_distr, l_distr))

        if "numberbatch" in self._config["scoring"]:
            nodes_embedding_norm = nodes_embedding / nodes_embedding.norm(2, dim = -1, keepdim=True)
            input_embedding_t = input_embedding_t / input_embedding_t.norm(2, dim = -1, keepdim=True)
            label_embedding = label_embedding / label_embedding.norm(2, dim = -1, keepdim=True)

            in_distr = torch.matmul(input_embedding_t, nodes_embedding_norm.t())#.sum(1)#max(1)[0]
            with torch.no_grad():
                mean = in_distr.mean(1, keepdim=True)
                std = in_distr.std(1, keepdim=True)
                mask = (in_distr < mean - 2*std) | (in_distr> mean+ 2*std) #
            in_distr = (in_distr * mask).sum(1)

            with torch.no_grad():
                mean = in_distr.mean(1, keepdim=True)
                std = in_distr.std(1, keepdim=True)
                mask = (in_distr < mean - 2*std) | (in_distr> mean+ 2*std) #
            in_distr = (in_distr * mask)


            l_distr =  torch.mm(label_embedding, nodes_embedding_norm.t()) #
            # sim =  (in_distr)[:, None] * (l_distr)[None]
            sim = self._sim(in_distr, l_distr)
            # sim = torch.mm(in_distr, l_distr.t())
            # sim = torch.mm(in_distr.softmax(-1), l_distr.t().softmax(-1))
            # sim = torch.mm(
            #     in_distr/ in_distr.norm(p=2, dim=-1, keepdim=True),
            #          (l_distr / l_distr.norm(p=2, dim=-1, keepdim=True)).t() )

            l.append(sim)

            # l.append(sim.relu() / sim.relu().sum(-1, keepdim=True))

        # import matplotlib.pyplot as plt
        # plt.scatter(range(l_distr.shape[-1]),l_distr[0].cpu().detach())
        # plt.plot(self.adjencies[0])
        # plt.plot(l_distr[1].cpu().detach())
        # plt.plot(l_distr[2].cpu().detach())
        # plt.plot(l_distr[3].cpu().detach())
        # plt.plot((self.adjencies[0]>0).int().cpu().detach())
        # plt.show()

        scores = torch.stack(l,-1).mean(-1)



        if kw:
            return scores, in_distr
        return scores

    def words(self, x, n = 10):
        self.eval()
        with torch.no_grad():
            tok = self.transform(x)
            scores, graphkw = self.forward(tok, kw=True)
        for b, text in enumerate(tok["text"]):
            print(" ".join(text))
            cls = scores.argmax(-1)[b].item()
            print([self._node_list[i.item()] for i in (graphkw).topk(10, dim=-1)[1][b]])
            print(list(self.classes.keys())[cls])
            print("\n\n")

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