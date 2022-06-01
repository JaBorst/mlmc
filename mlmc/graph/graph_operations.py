import networkx as nx
import torch
import re


def cos(x,y):
    """
    Calculates cosine similarity between two tensors.

    :param x: Tensor
    :param y: Tensor
    :return: Cosine similarity
    """
    return torch.matmul((x/x.norm(p=2,dim=-1,keepdim=True)) , (y/y.norm(p=2,dim=-1,keepdim=True)).t())


def extend(classes, graph, depth=1, topk=20, allow_non_alignment=False, device="cpu"):
    """
    Return a subgraph of relevant nodes.
    Args:
        classes: the classes mapping
        graph:  The graph from which to draw the subgraph from
        depth: The longest path from a classes label to any node

    Returns: Subgraph
    """
    augmented = embed_align(classes, graph, "glove300", topk=topk, batch_size=64, device=device)
    if not allow_non_alignment:
        assert len(augmented) == len([x for x,v in augmented.items() if len(v) > 0]), \
            "Not every class could be aligned in the graph: "+ ", ".join([x for x,v in augmented.items() if len(v) == 0])
    subgraph = nx.Graph()
    subgraph.add_nodes_from(classes.keys())
    for key, nodes in augmented.items():
        for n in nodes:
            subgraph.add_edge(key,n)

    current_nodes = list(subgraph.nodes)[len(classes):]
    for i in range(1,depth):
        next_level = {x:graph.adj[x] for x in current_nodes}
        next_nodes = []
        for u, v_dict in next_level.items():
            for v, l in v_dict.items():
                next_nodes.append(v)
                subgraph.add_edge(u,v, label=l["label"])
        current_nodes = next_nodes

    # Add all edges of nodes that are both in the subgraph already.
    for node in subgraph.nodes():
        for k, v in graph.adj.get(node,{}).items():
            if k in subgraph.nodes:
                subgraph.add_edge(node, k, label=v["label"])
    return subgraph


import networkx as nx
import torch

def subgraphs(classes, graph, depth=1, model="glove50", topk=10,  allow_non_alignment=False, batch_size=50, device="cpu"):
    # TODO: Documentation
    from ..representation import Embedder

    e = Embedder(model, device=device, return_device=device)
    classes_tokens = [" ".join(re.split("[/ _.-]", x.lower())) for x in classes.keys()]

    class_embeddings = torch.stack([x.mean(-2) for x in e.embed(classes_tokens, None)],0)

    scores = []
    for batch in e.embed_batch_iterator(list(graph.nodes), batch_size=batch_size):
        scores.append(cos(class_embeddings, torch.stack([x.mean(-2) for x in batch],0)).t())
    scores = torch.cat(scores, 0)
    scores[torch.isnan(scores)] = 0

    similar_nodes = scores.topk(topk, dim=0)[1].t().cpu()
    augmented = {k:[list(graph.nodes)[x.item()] for x in v] for k,v in zip(classes.keys(), similar_nodes)}


    if not allow_non_alignment:
        assert len(augmented) == len([x for x,v in augmented.items() if len(v) > 0]), \
            "Not every class could be aligned in the graph: "+ ", ".join([x for x,v in augmented.items() if len(v) == 0])
    subgraph = nx.OrderedDiGraph()
    subgraph.add_nodes_from(classes.keys(), type="label")
    for key, nodes in augmented.items():
        for n in nodes:
            subgraph.add_node(n, **graph.nodes(True)[n])
            subgraph.add_edge(key,n)

    current_nodes = list(subgraph.nodes)[len(classes):]
    for i in range(1,depth):
        next_level = {x:graph.adj[x] for x in current_nodes}
        next_nodes = []
        for u, v_dict in next_level.items():
            for v, l in v_dict.items():
                next_nodes.append(v)
                subgraph.add_node(v, **graph.nodes(True)[v])
                subgraph.add_edge(u,v, label=l["label"])
        current_nodes = next_nodes

    # Add all edges of nodes that are both in the subgraph already.
    for node in subgraph.nodes():
        for k, v in graph.adj.get(node,{}).items():
            if k in subgraph.nodes:
                subgraph.add_edge(node, k, label=v["label"])
    return subgraph


def plot_activation(graph, classes, scores, tr, target=None, title = None, layout="lgl", options={}):
    """
    Draws a plot of a given graph.

    :param graph: A networkx graph
    :param classes: The classes mapping
    :param scores: TODO: Documentation
    :param tr: TODO: Documentation
    :param target: The target where the graph should be plotted (None/cairo.Surface/string)
    :param title: Title of the plot
    :param layout: Layout of the graph (see igraph.Graph.layout_*)
    :param options: Additional keyword arguments of the chosen layout
    :return: A plot
    """
    import igraph
    g = igraph.Graph(directed=True)
    g.add_vertices(list(graph.nodes()))
    g.add_edges(graph.edges())

    for x in g.vs:
        x["score"] = scores[x["name"]] -tr

    def score_color(alpha):
        """
        Get a color depending on the given alpha value. Used in conjunction with vertex sequence score.

        :param alpha: Alpha channel value (score)
        :return: RGBA values
        """
        if alpha <= 0:
            return igraph.color_name_to_rgb("red") + (-alpha,)
        else:
            return igraph.color_name_to_rgb("green") + (alpha,)

    import random
    random.seed(42)
    visual_style = {}
    visual_style["vertex_size"] = 5
    visual_style["vertex_color"] = [score_color(x) for x in g.vs["score"]]
    visual_style["vertex_label"] = [x["name"] if x["name"] in classes.keys() or x["score"]>0 else None for x in g.vs]
    visual_style["vertex_frame_color"] = [score_color(x)[:3] for x in g.vs["score"]]
    visual_style["vertex_frame_width"] = 0.1
    visual_style["vertex_label_size"] = 5
    visual_style["edge_width"] = 0.01
    # visual_style["edge_color"] = (0.1,0.1,0.1)
    # visual_style["label_size"] = 0.1
    # visual_style["edge_width"] = [1 + 2 * int(is_formal) for is_formal in g.es["is_formal"]]
    visual_style["layout"] =  g.layout(layout, **options)
    visual_style["bbox"] =  (1000, 1000)
    visual_style["margin"] = 20
    visual_style["edge_arrow_size"] = 0.1
    visual_style["edge_color"]=  "grey"#"(0.1,0.1,0.1)

    if title is not None:
        visual_style["main"] = title

    if target is not None:
        igraph.plot(g,target, **visual_style)
    else:
        return igraph.plot(g, **visual_style)

from tqdm import tqdm
import requests
import string
def augment_wikiabstracts(graph):
    """
    Retrieves the Wikipedia abstract corresponding to each Wikidata entry and adds it to the graph.

    :param graph: A networkx graph
    :return: Graph with added Wikipedia abstracts.
    """
    abstracts = {}
    batch_size = 20
    for ind in tqdm(list(range(0, len(graph), batch_size))):
        batch = list(graph)[ind:(ind + batch_size)]
        batch_mapping = {n:"".join([i for i in n.replace("&amp;"," and ") if i in string.ascii_letters+" &"]).split("  ")[-1] for n in batch}
        url = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles=" + "|".join(batch_mapping.values())
        r = requests.get(url)
        data = r.json()
        mappings = {}
        if "redirects" in data["query"]:
            mappings = {x["from"]:x["to"] for x in data["query"]["redirects"]}

        extract_dict = {v["title"]:{"extract": v["extract"], "pageid": v["pageid"]} for k,v in data["query"]["pages"].items() if int(k)>0}
        for b in batch:
            if mappings.get(batch_mapping[b],batch_mapping[b]) in extract_dict.keys():
                abstracts[b] = extract_dict.get(mappings.get(batch_mapping[b],batch_mapping[b]))
    nx.set_node_attributes(graph, abstracts)
    return graph

def ascii_graph(graph):
    """
    Removes all non-ASCII characters from a graph.

    :param graph: A networkx graph
    :return: Cleaned graph
    """
    import string
    from copy import deepcopy
    def char_string(text):
        """
        Removes all non-ASCII characters from a string and replaces them with whitespaces.

        :param text: A string
        :return: Cleaned string
        """
        if isinstance(text, int) or isinstance(text, float): text = str(text)
        if not isinstance(text, str): return text
        return ''.join([i if i in string.ascii_letters+string.digits+" " else " " for i in text ])
    def char_dict(d):
        """
        Removes all non-ASCII characters from every dictionary tuple and replaces them with whitespaces.

        :param d: A dictionary
        :return: Cleaned dictionary
        """
        return {char_string(k):char_string(v) for k, v in d.items()}

    new_graph = deepcopy(graph)
    new_graph = nx.relabel.relabel_nodes(new_graph, {x:char_string(x) for x in graph.nodes()})

    nx.set_node_attributes(new_graph,
                        {x:{char_string(k):char_string(v) if not isinstance(v, dict) else char_dict(v) for k,v in att.items()} for x,att in dict(new_graph.nodes(True)).items()}
                             )
    return new_graph
