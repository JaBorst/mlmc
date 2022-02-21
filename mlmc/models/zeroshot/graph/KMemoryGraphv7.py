import networkx as nx

from mlmc.models.abstracts.abstracts_zeroshot import TextClassificationAbstractZeroShot
from mlmc.models.abstracts.abstract_sentence import SentenceTextClassificationAbstract
import torch
from mlmc.modules.dropout import VerticalDropout
from mlmc.graph import get as gget
from mlmc.modules.module_tfidf import TFIDFAggregation
import networkx as nx
from datasketch import MinHashLSH, MinHash
from tqdm import tqdm
# helper for
def merge_nodes(graph,classes, num_perm=32, threshold=0.9, n = 3, ):
    def _mh(x, k):
        x = x.upper()
        k = k if isinstance(k, (tuple, list)) else [k]
        def ngrams(x, k):
            return [x[i:(i + k)] for i in range(len(x) - k + 1)]
        m1 = MinHash(num_perm)
        for kk in k:
            for i in ngrams(x, kk): m1.update(i.encode("utf8"))
        for w in x.split(" "):
            # m1.update(w.encode("utf8"))
            for kk in k:
                for i in ngrams(w, kk): m1.update(i.encode("utf8"))
        return m1

    lsh = MinHashLSH(threshold=0.95, num_perm=num_perm)
    for x in tqdm(graph.nodes): lsh.insert(x, _mh(x, n))
    sims = {k: lsh.query(_mh(k, n)) for k in graph.nodes}
    resultsets = [set([k] + v) for k, v in sims.items()]

    len_2 = len(resultsets)
    len_1 = len_2 + 1
    iterations = 0
    while len_1 != len_2:
        iterations += 1
        resultsets = set([frozenset(list(k) + sum([sims[v] for v in k], [])) for k in resultsets])
        len_1 = len_2
        len_2 = len(resultsets)

    resultsets = [list(x) for x in resultsets]

    relabel_dict = {}
    for s in [x for x in resultsets if len(x) > 1]:
        i = 0 if not any([x in classes for x in s]) else [s.index(x)  for x in classes if x in s][0]
        for j, k in enumerate(s):
            if i==j: pass
            else: relabel_dict[k] = s[i]
    return relabel_dict


class KMemoryGraph(SentenceTextClassificationAbstract, TextClassificationAbstractZeroShot):
    def __init__(self, similarity="cosine", dropout=0.5, entropy=True,
                 measures=None,depth=1,fallback_classifier = False, rp = False, rp_trainable=False, inner_dim=32, n_proj=4,
                 graph="wordnet", *args, **kwargs):
        super(KMemoryGraph, self).__init__(*args, **kwargs)
        if measures is None:
            measures= ["pooled_similarity", "keyword_similiarity_mean"]

        if fallback_classifier:
            measures = measures + ["fallback_classifier"]

        self._config["fallback_classifier"] = fallback_classifier
        self.dropout = torch.nn.Dropout(dropout)

        self.entailment_projection = torch.nn.Linear(3 * self.embeddings_dim, self.embeddings_dim)
        self.entailment_projection2 = torch.nn.Linear(self.embeddings_dim, 1)


        self._config["dropout"] = dropout
        self._config["similarity"] = similarity
        self._config["entropy"] = entropy
        self.agg = TFIDFAggregation()

        self.set_scoring(measures)
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
        graph = gget(self._config["graph"])#.to_undirected()


        nodes = [sum([self.map[k] for k in self.classes.keys()],[])]
        for i in range(1, self._config["depth"]):
            nodes.append(list(set(sum([list(graph.neighbors(x)) for x in nodes[i-1] if x in graph][:200],[]))))
        nodes = sum(nodes,[])
        nodes = [x for x in nodes if x.count("_") < 3 and ":" not in x and len(x) > 3]
        g = graph.subgraph(nodes).copy()
        g.add_edges_from([(k, v) for k in self.classes.keys() for v in self.map[k]])
        g.add_edges_from([( v,k) for k in self.classes.keys() for v in self.map[k]])
        g.remove_nodes_from([node for node, degree in dict(g.degree()).items() if degree < 2])
        g =nx.relabel_nodes(g,merge_nodes(g, threshold=0.7, classes=self.classes.keys()))
        return g

    def update_memory(self):
        """
        Method to change the current target variables
        Args:
            classes: Dictionary of class mapping like {"label1": 0, "label2":1, ...}

        Returns:

        """
        self.g = self._get_graph()
        # self.g = self.g.to_undirected()
        self.build()

        self._node_list = sorted(list(self.g.nodes))
        assert all([cls in self._node_list for cls in self.classes.keys()])
        self.nodes = self.transform([x.replace("_", " ") for x in self._node_list])


        self._class_nodes = {k:self._node_list.index(k) for k in self.classes.keys()}
        adj = nx.adj_matrix(self.g, self._node_list)
        adj = torch.FloatTensor(adj.toarray())
        self.adjacency_shape = adj.shape
        with torch.no_grad():
            for _ in range(self._config["depth"]-1):
                adj = adj / (adj.sum(-1, keepdim=True)+1e-10)
                adj = torch.mm(adj.t(),adj)
        adj= adj/ adj.sum(-1, keepdim=True)
        self.adjencies = torch.nn.Parameter(torch.stack([adj[i] for i in self._class_nodes.values()], 0).float()).to(self.device).detach()
        self.adj = torch.nn.Parameter(adj).to_sparse().to(self.device)


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
            r = torch.matmul(x,y.t())#(x.unsqueeze(-2) * y.unsqueeze(-3) ).sum(-1)
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



        input_embedding=input_embedding_t
        nodes_embedding = self._mean_pooling(nodes_embedding, self.nodes["attention_mask"])
        input_embedding = self._mean_pooling(input_embedding, x["attention_mask"])
        label_embedding = self._mean_pooling(label_embedding, self.label_dict["attention_mask"])

        pooled_similarity = self._sim(input_embedding, label_embedding).squeeze(-1)  # pooled-similarity
        nodes_embedding_norm = nodes_embedding / nodes_embedding.norm(2, dim = -1, keepdim=True)
        input_embedding_t = input_embedding_t / input_embedding_t.norm(2, dim = -1, keepdim=True)
        label_embedding = label_embedding / label_embedding.norm(2, dim = -1, keepdim=True)



        in_distr = torch.matmul(input_embedding_t, nodes_embedding_norm.t())#.sum(1)#max(1)[0]
        # l_distr = torch.matmul(label_embedding, nodes_embedding_norm.t())#.sum(1)#max(1)[0]
        t_distr = torch.matmul(input_embedding_t, label_embedding.t())#.sum(1)#max(1)[0]

        with torch.no_grad():
            mean = (in_distr.mean((-1,-2), keepdim=True)).float()
            std = (in_distr.std((-1,-2), keepdim=True)).float()
            mask = (in_distr > (mean + 2*std))#.to_sparse()
            topmask = torch.nn.functional.one_hot(in_distr.topk(dim=-1, k=5)[1],len(self._node_list)).sum(-2)
            # topmask = torch.nn.functional.one_hot(in_distr.argmax(-1),len(self._node_list))#.sum(-2)
            mask = (mask * topmask * x["attention_mask"].unsqueeze(-1)).float()#.to_sparse()
            gumbel = mask - in_distr
        # mask = in_distr + gumbel

        in_distr = (in_distr*mask).sum(1)

        # in_distr[:, None] * self.adjencies[None]
        # kg_distr = torch.matmul(mask/(mask.sum(-1, keepdim=True)+1e-12), nodes_embedding)
        # kg_distr = self._mean_pooling(kg_distr, x["attention_mask"])
        # pooled_similarity2 = self._sim(kg_distr, label_embedding).squeeze(-1)  # pooled-similarity


        sim=( in_distr[:, None] * self.adjencies[None]).sum(-1) #(mask.sum([1,2]).unsqueeze(-1))*

        with torch.no_grad():
            t_mean = (t_distr.mean((1,2), keepdim=True)).float()
            t_std = (t_distr.std((1,2), keepdim=True)).float()
            t_mask = (t_distr > (t_mean +  2*t_std))  # .to_sparse()
            t_topmask = torch.nn.functional.one_hot(t_distr.argmax(-1), len(self.classes))
            t_mask = t_mask* t_topmask * x["attention_mask"].unsqueeze(-1)
            t_mask.sum(1)
            gumbel2 = t_mask - t_distr
        # t_mask = t_distr + gumbel2
        sim3 = (t_distr*t_mask).sum(1) #/ (t_mask.sum(1) + 1e-6)
        # sim4 = ((t_mask).sum(1)) / x["attention_mask"].sum(-1,keepdim=True) #/ ((mask_neg[:, None] * self.adjencies[None]).sum(-1))

        # all =torch.stack([sim.log_softmax(-1), sim3.log_softmax(-1), pooled_similarity.log_softmax(-1)],-1).mean(-1)
        all =torch.stack([(0.5*(1+sim)).log(), (0.5*(1+sim3)).log(), pooled_similarity],-1).mean(-1)#.log_softmax(-1)
        # all = pooled_similarity + pooled_similarity2

        # labels = [['Business'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sci/Tech'], ['Sports'], ['Sports'], ['Sports'], ['Sports'], ['Sports'], ['Sports'], ['World'], ['World'], ['World'], ['World'], ['World'], ['World'], ['World'], ['World'], ['Sports'], ['Business'], ['World'], ['Sci/Tech'], ['Sports'], ['Sports'], ['World'], ['Sci/Tech'], ['World'], ['Sports']]
        # i=-1
        # i=i+1
        # print(x["text"][i])
        # print("sim :",sim[i],sim[i].argmax(-1).item())
        # # print("sim2:",sim2[i],sim2[i].argmax(-1).item())
        # print("sim3:",sim3[i],sim3[i].argmax(-1).item())
        # # print("sim4:",sim4[i],sim4[i].argmax(-1).item())
        # print(pooled_similarity[i],pooled_similarity[i].argmax(-1).item())
        # print(all[i], all[i].argmax(-1).item())
        # print(labels[i])
        # # print((sim2[i]).log_softmax(-1) + pooled_similarity[i])
        # # print((sim[i]+sim2[i]).log_softmax(-1) + pooled_similarity[i])
        # print([self._node_list[k] for k in (in_distr[i]* self.adjencies[all[i].argmax(-1).item()]).topk(20)[1]])
        # print([self._node_list[k] for k in in_distr[i].topk(20)[1]])
        # print([self._node_list[k] for k in torch.where(mask.sum(1)[i]>0)[0]])
        # print([x["text"][i][k] for k in torch.where(t_mask[i][...,all[i].argmax(-1).item()]>0)[0] if k<len(x["text"][i])-1])
        # print(self.classes)
        # import matplotlib.pyplot as plt
        # plt.scatter(range(l_distr.shape[-1]),l_distr[0].cpu().detach())
        # plt.plot(self.adjencies[0])
        # plt.plot(l_distr[1].cpu().detach())
        # plt.plot(l_distr[2].cpu().detach())
        # plt.plot(l_distr[3].cpu().detach())
        # plt.plot((self.adjencies[0]>0).int().cpu().detach())
        # plt.show()

        scores = all#(sim+sim2).log_softmax(-1) + pooled_similarity



        if kw:
            return scores, in_distr, mask
        return scores

    def words(self, x, n = 10):
        self.eval()
        with torch.no_grad():
            tok = self.transform(x)
            scores, graphkw, t_mask = self.forward(tok, kw=True)
        for b, text in enumerate(tok["text"]):
            print(" ".join(text))
            cls = scores.argmax(-1)[b].item()
            print([self._node_list[i.item()] for i in (graphkw).topk(10, dim=-1)[1][b]])
            print(list(self.classes.keys())[cls])
            print("\n\n")
        for b, (text, tkw) in enumerate(zip(tok["text"],t_mask)):
            for i, (token, tk) in enumerate(zip(text,tkw[1:])):
                print(token, [self._node_list[x.item()] for x in torch.where(tk)[0]])

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