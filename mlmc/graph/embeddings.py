# https://github.com/shenweichen/GraphEmbedding

from sklearn.decomposition import NMF
from node2vec import Node2Vec
import numpy
import networkx as nx

def get_node2vec(adjacency, dim):
    graph = nx.from_numpy_matrix(adjacency)
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=dim, walk_length=30, num_walks=500, workers=4)
    model = node2vec.fit(window=50, min_count=1, batch_words=4)
    return model.wv.vectors

def get_nmf(adjacency, dim):
    model = NMF(n_components=dim, init='random', random_state=0)
    W = model.fit_transform(adjacency)
    H = model.components_
    return W
