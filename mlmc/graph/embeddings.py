# https://github.com/shenweichen/GraphEmbedding

from sklearn.decomposition import NMF
from sklearn import random_projection
from node2vec import Node2Vec

import networkx as nx

def get_node2vec(X, dim):
    graph = nx.from_numpy_matrix(X)
    node2vec = Node2Vec(graph, dimensions=dim, walk_length=30, num_walks=500, workers=4)
    model = node2vec.fit(window=50, min_count=1, batch_words=4)
    return model.wv.vectors

def get_nmf(X, dim):
    model = NMF(n_components=dim, init='random', random_state=0)
    W = model.fit_transform(X)
    H = model.components_
    return W

def get_random_projection(X, dim):
    transformer = random_projection.GaussianRandomProjection(n_components=dim)
    W = transformer.fit_transform(X)
    return W
