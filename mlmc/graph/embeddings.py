# https://github.com/shenweichen/GraphEmbedding

import networkx as nx
import numpy as np
from sklearn import random_projection
from sklearn.decomposition import NMF


def subgraph_extract(X, graph, subnodelist):
    """
    Extracts a subset of node embeddings from a graph.

    :param X: Node embeddings of graph
    :param graph: A networkx graph
    :param subnodelist: Dictionary of nodes for which the embedding will be returned
    :return: Embeddings of all nodes in subnodelist
    """
    new = np.zeros_like(X)
    for i, nm in enumerate(graph.nodes):
        if nm in subnodelist.keys():
            new[subnodelist[nm]] = X[i]
    return new


def get_node2vec(graph, classes, dim, return_all=False):
    """
    Embed a graph using the node2vec algorithm
    :param graph: A networkx graph
    :param classes: Classes to be extracted from the graph. (The subset of nodes for which the embedding will be returned)
    :param dim: The dimension of the embedding
    :param return_all: If True the embedding to every node will be returned. If False only the subset of embeddings for nodes that are in 'classes' will be returned.
    :return: A Matrix of embedding vectors
    """
    from node2vec import Node2Vec
    node2vec = Node2Vec(graph, dimensions=dim, walk_length=30, num_walks=500, workers=4)
    model = node2vec.fit(window=50, min_count=1, batch_words=4)
    W = model.wv.vectors
    if return_all: W=subgraph_extract(W,graph, dict(zip(list(graph.nodes), range(len(graph.nodes)))))
    else: W=subgraph_extract(W,graph,classes)
    return W

def get_nmf(graph, classes, dim, return_all=False):
    """
    Embed a graph using the non-negative matrix factorization algorithm.
    :param graph: A networkx graph
    :param classes: Classes to be extracted from the graph. (The subset of nodes for which the embedding will be returned)
    :param dim: The dimension of the embedding
    :param return_all: If True the embedding to every node will be returned. If False only the subset of embeddings for nodes that are in 'classes' will be returned.
    :return: A Matrix of embedding vectors
    """
    model = NMF(n_components=dim, init='random', random_state=0)
    W = model.fit_transform(nx.to_numpy_array(graph))
    H = model.components_
    if return_all: W=subgraph_extract(W,graph, dict(zip(list(graph.nodes), range(len(graph.nodes)))))
    else: W=subgraph_extract(W,graph,classes)
    return W

def get_random_projection(graph, classes, dim, return_all=False):
    """
    Embed a graph using the random projection algorithm.
    :param graph: A networkx graph
    :param classes: Classes to be extracted from the graph. (The subset of nodes for which the embedding will be returned)
    :param dim: The dimension of the embedding
    :param return_all: If True the embedding to every node will be returned. If False only the subset of embeddings for nodes that are in 'classes' will be returned.
    :return: A Matrix of embedding vectors
    """
    transformer = random_projection.GaussianRandomProjection(n_components=dim)
    W = transformer.fit_transform(nx.to_numpy_array(graph))
    if return_all: W=subgraph_extract(W,graph, dict(zip(list(graph.nodes), range(len(graph.nodes)))))
    else: W=subgraph_extract(W,graph,classes)
    return W
