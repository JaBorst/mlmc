import networkx as nx
from mlmc.graph import get as gget
import torch
from .helpers import merge_nodes

class Graph(torch.nn.Module):
    def __init__(self, graph, map, depth=2):
        super(Graph, self).__init__()
        self._map = map
        self._graph = graph
        self._depth = depth

    def _get_graph(self, classes):
        graph = gget(self._graph)
        nodes = [sum([self.map[k] for k in self.classes.keys()], [])]
        for i in range(1, self._config["depth"]):
            nodes.append(list(set(sum([list(graph.neighbors(x)) for x in nodes[i - 1] if x in graph][:200], []))))
        nodes = sum(nodes, [])
        nodes = [x for x in nodes if x.count("_") < 3 and ":" not in x and len(x) > 3]
        g = graph.subgraph(nodes).copy()
        g.add_edges_from([(k, v) for k in classes.keys() for v in self.map[k]])
        g.add_edges_from([(v, k) for k in classes.keys() for v in self.map[k]])
        g.remove_nodes_from([node for node, degree in dict(g.degree()).items() if degree < 2])
        g = nx.relabel_nodes(g, merge_nodes(g, threshold=0.7, classes=classes.keys()))
        return g