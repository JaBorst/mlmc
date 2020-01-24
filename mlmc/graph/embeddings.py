# https://github.com/shenweichen/GraphEmbedding

from sklearn.decomposition import NMF
from sklearn import random_projection
from node2vec import Node2Vec
import numpy as np

import networkx as nx
def subgraph_extract(X, graph, subnodelist):
    new = np.zeros_like(X)
    for i, nm in enumerate(graph.nodes):
        if nm in subnodelist.keys():
            new[subnodelist[nm]] = X[i]
    return new


def get_node2vec(graph, classes, dim, return_all=False):
    node2vec = Node2Vec(graph, dimensions=dim, walk_length=30, num_walks=500, workers=4)
    model = node2vec.fit(window=50, min_count=1, batch_words=4)
    W = model.wv.vectors
    if return_all: W=subgraph_extract(W,graph, dict(zip(graph.nodes, len(range(graph.nodes)))))
    else: W=subgraph_extract(W,graph,classes)
    return W

def get_nmf(graph, classes, dim, return_all=False):
    model = NMF(n_components=dim, init='random', random_state=0)
    W = model.fit_transform(nx.to_numpy_array(graph))
    H = model.components_
    if return_all: W=subgraph_extract(W,graph, dict(zip(graph.nodes, len(range(graph.nodes)))))
    else: W=subgraph_extract(W,graph,classes)
    return W

def get_random_projection(graph, classes, dim, return_all=False):
    transformer = random_projection.GaussianRandomProjection(n_components=dim)
    W = transformer.fit_transform(nx.to_numpy_array(graph))
    if return_all: W=subgraph_extract(W,graph, dict(zip(graph.nodes, len(range(graph.nodes)))))
    else: W=subgraph_extract(W,graph,classes)
    return W
