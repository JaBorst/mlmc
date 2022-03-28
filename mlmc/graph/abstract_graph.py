import networkx as nx
from mlmc.graph.graph_loaders import get as gget
import torch
from sklearn.manifold import TSNE
from datasketch import MinHashLSH, MinHash
from tqdm import tqdm

class Graph(torch.nn.Module):
    def __init__(self, graph,  map=None, matcher="fuzzy", depth=2, calc_depth=True, distances=False):
        super(Graph, self).__init__()
        self._config={
            "graph": graph,
            "depth": depth,
            "matcher": matcher,
            "calc_depth": calc_depth,
            "distances": distances,
        }
        self.map = map


    def _get_map_graph(self):
        graph = gget(self._config["graph"])
        nodes = [sum([self.map.get(k,[k]) for k in self.classes.keys()], [])]
        for i in range(1,  self._config["depth"]):
            nodes.append(list(set(sum([list(graph.neighbors(x)) for x in nodes[i - 1] if x in graph][:200], []))))
        nodes = sum(nodes, [])
        nodes = [x for x in nodes if x.count("_") < 3 and ":" not in x and len(x) > 3]
        g = graph.subgraph(nodes).copy()
        g.add_edges_from([(k, v) for k in self.classes.keys() for v in self.map.get(k,[k])])
        g.add_edges_from([(v, k) for k in self.classes.keys() for v in self.map.get(k,[k])])
        g.remove_nodes_from([node for node, degree in dict(g.degree()).items() if degree < 2])
        g = nx.relabel_nodes(g, self._merge_nodes(g, threshold=0.7, classes=self.classes.keys()))
        return g

    def _get_match_graph(self):
        from fuzzylists import RIList
        base_graph = gget(self._config["graph"])
        l1 = RIList(base_graph.nodes)
        l2 = [RIList(x.split(","),threshold=0.9) for x in self.classes.keys()]

        # l2 = RIList([x.lower() for x in self.classes.keys()],threshold=0.85)
        mapping = [sum([[list(base_graph.nodes)[n] for n in x] for x in l.map(l1)],[]) for l in l2]
        mapping = list(zip(self.classes.keys(),mapping))
        print("Found class mapping:", mapping)
        base_graph.add_nodes_from(self.classes.keys())
        base_graph.add_edges_from([(k,e) for k,v in mapping for e in v])

        retain = []
        for n in tqdm(base_graph.nodes):
            for cls in self.classes.keys():
                try:
                    l = (len(nx.shortest_path(base_graph, cls, n)))
                except nx.exception.NetworkXNoPath:
                    l = (100)
                if l<self._config["depth"]+1:
                    retain.append(n)
                    continue
        g = nx.subgraph(base_graph, list(self.classes.keys()) + retain)
        raise g

    def fit(self, classes):
        """Generate a subgraph fitted to the current set of classes"""
        self.classes = classes
        if self.map is not None:
            self._g = self._get_map_graph()
        else:
            self._g = self._get_match_graph()

    def show_graph(self, vectors, classes):
        """
        Plot A TSNE
        :param vectors: Vectors to plot
        :param classes: Labels of the vectors to plot
        :return:
        """
        import matplotlib.pyplot as plt
        X_embedded = TSNE(n_components=2).fit_transform(vectors)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1])
        for i, txt in enumerate(classes.keys()):
            ax.annotate(txt, (X_embedded[i, 0], X_embedded[i, 1]))
        plt.show()

    def _merge_nodes(self, graph,classes, num_perm=32, threshold=0.95, n = 3):
        """Merge nodes where the names of the nodes are (almost) the same."""
        def _mh(x, k):
            x = x.upper()
            k = k if isinstance(k, (tuple, list)) else [k]
            def ngrams(x, k):
                return [x[i:(i + k)] for i in range(len(x) - k + 1)]
            m1 = MinHash(num_perm)
            for kk in k:
                for i in ngrams(x, kk): m1.update(i.encode("utf8"))
            for w in x.split(" "):
                for kk in k:
                    for i in ngrams(w, kk): m1.update(i.encode("utf8"))
            return m1
        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
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


    def _create_vars(self):

        #Sorted nodelist for safety
        self._node_list = sorted(list(self._g.nodes))
        assert all([cls in self._node_list for cls in self.classes.keys()])

        # Identify the classnodes.
        self._class_nodes_index = {k:self._node_list.index(k) for k in self.classes.keys()}


        adj = nx.adj_matrix(self._g, self._node_list)
        adj = torch.FloatTensor(adj.toarray())

        if self._config["calc_depth"]:
            with torch.no_grad():
                for _ in range(self._config["depth"]-1):
                    adj = adj / (adj.sum(-1, keepdim=True)+1e-10)
                    adj = torch.mm(adj.t(),adj)
            adj= adj/ (adj.sum(-1, keepdim=True) +1e-10)

        self._adjacency = torch.nn.Parameter(adj).to_sparse()
        self.shape = adj.shape
        self._class_adjacency = torch.nn.Parameter(torch.stack([adj[i] for i in self._class_nodes_index.values()], 0).float()).to_sparse().detach()

        if self._config["distances"]:
            l = {}
            for cls in self.classes.keys():
                l[cls] = []
                for n in self._node_list:
                    try:
                        l[cls].append(len(nx.shortest_path(self._g, cls, n)))
                    except nx.exception.NetworkXNoPath:
                        l[cls].append(0)

            d = 1/(torch.tensor(list(l.values()))-1)
            d = d.clamp(0,1)
            self._distance = torch.nn.Parameter(d).to_sparse()


    def __call__(self, classes):
        self.fit(classes)
        self._create_vars()

    def get_adjacency(self):
        return self._adjacency

    def get_class_adjacency(self):
        return self._class_adjacency

    def get_class_nodes_index(self):
        return self._class_nodes_index

    def get(self, device="cpu"):
        if self._config["distances"]:
            return self._node_list, \
                   self._class_nodes_index, \
                   self._adjacency.to(device), \
                   self._class_adjacency.to(device), \
                   self._distance.to(device)
        else:
            return self._node_list, \
                   self._class_nodes_index, \
                   self._adjacency.to(device), \
                   self._class_adjacency.to(device)
# from mlmc.graph.helpers import keywordmap
# g = Graph("wordnet", map=None)
# g(classes={"Sci/Tech":3, "World":0,"Sports":1, "Business":2})
