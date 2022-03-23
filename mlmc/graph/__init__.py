"""
Provides functions for loading predefined graphs
"""
from .helpers import cooc_matrix, merge_nodes
from .graph_loaders import load_wordnet, load_wordnet_sample, load_NELL,load_elsevier,load_conceptNet, load_stw, load_nasa, \
    load_gesis, load_mesh, load_afinn
from .graph_operations import subgraphs, embed_align, augment_wikiabstracts,ascii_graph
from .graph_operations import plot_activation
from .graph_insert import graph_insert_labels
register = {
    "wordnet": load_wordnet,
    "stw": load_stw,
    "nasa": load_nasa,
    "gesis": load_gesis,
    "mesh": load_mesh,
    "conceptnet": load_conceptNet,
    "elsevier": load_elsevier,
    "afinn": load_afinn

}


def get_graph(name: str):
    """
    Loads a graph.

    :param name: Name of the graph (see register.keys())
    :return: Function call of the chosen graph
    """
    fct = register.get(name)
    if fct is None:
        raise FileNotFoundError
    else:
        return fct()

import networkx as nx
def get(name: [list, str]):
    """
    Loads a graph. If multiple names are provided the union of the graphs is returned.

    :param name: Name(s) of the graph(s) to compose (see register.keys())
    :return: Merged graph
    """
    if isinstance(name, str):
        name = [name]
    return nx.compose_all([get_graph(x) for x in name])
