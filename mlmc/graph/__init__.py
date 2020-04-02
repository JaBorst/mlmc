from .helpers import cooc_matrix
from .embeddings import get_nmf, get_node2vec, get_random_projection
from .graph_loaders import load_wordnet, load_wordnet_sample, load_NELL
from .graph_loaders_elsevir import subgraph, load_elsevier, load_stw, load_nasa, \
    load_gesis, load_mesh, augment, sim_align, embed_align

register = {
    "wordnet": load_wordnet,
    "stw": load_stw,
    "nasa": load_nasa,
    "gesis": load_gesis,
    "mesh": load_mesh,
    "elsevier": load_elsevier

}

def get_graph(name: str):
    fct = register.get(name)
    if fct is None:
        raise FileNotFoundError
    else:
        return fct()

import networkx as nx
def get(name: [list, str]):
    if isinstance(name, str):
        name = [name]
    return nx.compose_all([get_graph(x) for x in name])


