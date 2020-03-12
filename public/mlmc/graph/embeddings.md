Module mlmc.graph.embeddings
============================

Functions
---------

    
`get_nmf(graph, classes, dim, return_all=False)`
:   Embed a graph using the non-negative matrix factorization algorithm.
    :param graph: A networkx graph
    :param classes: Classes to be extracted from the graph. (The subset of nodes for which the embedding will be returned)
    :param dim: The dimension of the embedding
    :param return_all: If True the embedding to every node will be returned. If False only the subset of embeddings for nodes that are in 'classes' will be returned.
    :return: A Matrix of embedding vectors

    
`get_node2vec(graph, classes, dim, return_all=False)`
:   Embed a graph using the node2vec algorithm
    :param graph: A networkx graph
    :param classes: Classes to be extracted from the graph. (The subset of nodes for which the embedding will be returned)
    :param dim: The dimension of the embedding
    :param return_all: If True the embedding to every node will be returned. If False only the subset of embeddings for nodes that are in 'classes' will be returned.
    :return: A Matrix of embedding vectors

    
`get_random_projection(graph, classes, dim, return_all=False)`
:   Embed a graph using the random projection algorithm.
    :param graph: A networkx graph
    :param classes: Classes to be extracted from the graph. (The subset of nodes for which the embedding will be returned)
    :param dim: The dimension of the embedding
    :param return_all: If True the embedding to every node will be returned. If False only the subset of embeddings for nodes that are in 'classes' will be returned.
    :return: A Matrix of embedding vectors

    
`subgraph_extract(X, graph, subnodelist)`
: