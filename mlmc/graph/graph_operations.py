import networkx as nx
import torch

def embed_align(classes, graph, model="glove50", topk=10, batch_size=50, device="cpu"):
    from ..representation import Embedder
    import re
    e = Embedder(model, device=device, return_device=device)
    classes_tokens = [" ".join(re.split("[/ _-]", x.lower())) for x in classes.keys()]

    class_embeddings = torch.stack([x.mean(-2) for x in e.embed(classes_tokens, None)],0)

    from ..representation.similarities import cos
    scores = []
    for batch in e.embed_batch_iterator(list(graph.nodes), batch_size=batch_size):
        scores.append(cos(class_embeddings, torch.stack([x.mean(-2) for x in batch],0)).t())

    scores = torch.cat(scores, 0)
    similar_nodes = scores.topk(topk, dim=0)[1].t().cpu()
    return {k:[list(graph.nodes)[x.item()] for x in v] for k,v in zip(classes.keys(), similar_nodes)}


def extend(classes, graph, depth=1, topk=20, allow_non_alignment=False, device="cpu"):
    """
    Return a subgraph of relevant nodes.
    Args:
        classes: the classes mapping
        graph:  The graph from which to draw the subgraph from
        depth: The longest path from a classes label to any node

    Returns:
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
    from ..representation import Embedder
    import re

    e = Embedder(model, device=device, return_device=device)
    classes_tokens = [" ".join(re.split("[/ _-]", x.lower())) for x in classes.keys()]

    class_embeddings = torch.stack([x.mean(-2) for x in e.embed(classes_tokens, None)],0)

    from ..representation.similarities import cos
    scores = []
    for batch in e.embed_batch_iterator(list(graph.nodes), batch_size=batch_size):
        scores.append(cos(class_embeddings, torch.stack([x.mean(-2) for x in batch],0)).t())

    scores = torch.cat(scores, 0)
    similar_nodes = scores.topk(topk, dim=0)[1].t().cpu()
    augmented = {k:[list(graph.nodes)[x.item()] for x in v] for k,v in zip(classes.keys(), similar_nodes)}

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


    matrix = e.embed(list(subgraph.nodes), pad=4)

    return subgraph, matrix


