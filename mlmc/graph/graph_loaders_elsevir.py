
import networkx as nx
import tempfile
from urllib.request import urlopen
import gzip
from mlmc.data.data_loaders import _save_to_tmp, _load_from_tmp
from rdflib import Graph as RDFGraph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import networkx as nx
import gzip
from tqdm import tqdm
import rdflib
from zipfile import ZipFile
from io import BytesIO

def transform(x, rg, lang="en"):
    if isinstance(x, rdflib.term.URIRef):
        from rdflib.namespace import SKOS
        if str(SKOS) in str(x): return [x.split("#")[-1]]
        else:
            return [str(l[1]) for l in rg.preferredLabel(x) if l[1].language==lang]
    elif isinstance(x, rdflib.term.Literal):
        if x.language != lang: return None
        else: return [str(x)]
    elif isinstance(x, rdflib.term.BNode):
        return None#[str(x)]
    return (x)


def transform_triples(rg,lang="en"):
    new_list =  [[transform(x, rg, lang) for x in t] for t in tqdm(rg)]
    new_list = [x for x in new_list
                if (not (any([r==[] for r in x]) or any([r is None for r in x]))) and ( "prefLabel" not in x[1])]
    return new_list



def load_mesh():

    url = "ftp://ftp.nlm.nih.gov/online/mesh/rdf/2020/"
    resp = urlopen(url+"mesh2020.nt.gz")
    f = gzip.open(resp, mode="rb")
    content = f.read().decode()
    rg = RDFGraph()
    with tempfile.TemporaryDirectory() as tmpdir:
        print(tmpdir)
        with open(tmpdir + "/tst.nt", "w") as f: f.write(content)
        rg = rg.parse(tmpdir + "/tst.nt", format='nt')

    G = rdflib_to_networkx_graph(rg)
    print("rdflib Graph loaded successfully with {} triples".format(len(rg)))
    return G

def load_nasa():
    data = _load_from_tmp("nasa")
    if data is not None:
        return data
    else:
        url = "https://www.sti.nasa.gov/download/NASA_Thesaurus_Alpha_SKOS.xml"
        response = urlopen(url)
        content = response.read().decode("utf-16")
        rg = RDFGraph()
        rg = rg.parse(data=content)
        literal_triples = transform_triples(rg)
        g = nx.DiGraph()
        for triple in literal_triples:
            for u in triple[0]:
                for v in triple[2]:
                    for p in triple[1]:
                        g.add_edge(u, v, label=p)
        _save_to_tmp("nasa", g)
        return g



def load_gesis():
    data = _load_from_tmp("gesis")
    if data is not None:
        return data
    else:
        url = "http://lod.gesis.org/thesoz-komplett.xml.gz"
        response = urlopen(url)
        f = gzip.open(response, mode="rb")
        content = f.read().decode()
        rg = RDFGraph()
        rg = rg.parse(data = content)
        literal_triples = transform_triples(rg)
        g = nx.DiGraph()
        for triple in literal_triples:
            for u in triple[0]:
                for v in triple[2]:
                    for p in triple[1]:
                        g.add_edge(u, v, label = p)
        _save_to_tmp("gesis",g)
        return g

def load_stw():
    data = _load_from_tmp("stw")
    if data is not None:
        return data
    else:
        url = "https://aspra29.informatik.uni-leipzig.de:9090/stw.rdf.zip"
        response = urlopen(url)
        zf = ZipFile(BytesIO(response.read()))
        with zf.open("stw.rdf") as f:
            content = f.read()
        rg = RDFGraph()
        rg = rg.parse(data=content)
        literal_triples = transform_triples(rg)
        g = nx.DiGraph()
        for triple in literal_triples:
            for u in triple[0]:
                for v in triple[2]:
                    for p in triple[1]:
                        g.add_edge(u, v, label=p)
        _save_to_tmp("stw",g)
        return g

def load_elsevier():
    """https://www.elsevier.com/__data/assets/pdf_file/0010/175249/ACAD_FP_FS_FPEFactSheet2019_WEB.pdf"""
    print("Be Careful. This is not yet the full elsevier thesaurus. ONly gesis, nasa and stw")
    gesis = load_gesis()
    nasa = load_nasa()
    stw = load_stw()
    c = nx.compose(nx.compose(gesis, nasa), stw)
    return c


import re
import string
import torch



def augment(classes, graph):
    def ortho_variant(x, y):
        return any([
            x in y,
            x.replace("sation", "zation") in y,
            x.replace("zation", "sation") in y,
            x.replace("ize", "ise") in y,
            x.replace("ise", "ize") in y,
            x.replace("yse", "yze") in y,
            x.replace("yze", "yse") in y,
        ])

    # see if the class name occurs in the graph in full
    augmented = {label:[x for x in graph.nodes if ortho_variant(label.lower(), x.lower()) ] for label in classes.keys()}
    # If not check for words with different ending ("especially: plural  "-s" and "-ies-> -y"
    augmented = {k:v if len(v)>0 else
                [x  for x in graph.nodes if  ortho_variant(k[:-1].lower(), x.lower()) or ortho_variant(k[:-3].lower(), x.lower())]
                 for k,v in augmented.items()}
    # If not check for tokens
    augmented = {label:v
                    if len(v)>0
                    else [x for x in graph.nodes
                          for w in re.sub("&/, -","","".join([x for x in label if x in string.ascii_letters+" "]).replace("  ", " ").lower()).split(" ")
                          if w.strip() == x and len(w.strip()) != 0 ]  for label,v in augmented.items()}

    augmented = {
        label: v if len(v) > 0 else [x for x in graph.nodes for w in re.split("[/, &-]{1,2}", label.lower()) if ortho_variant(w[:-1], x) or ortho_variant(w[:-3], x)] for
        label, v in augmented.items()}

    augmented = {k: list(set(v)) for k, v in augmented.items()}
    return augmented

def sim_align(classes, graph):
    # raise NotImplementedError
    import mlmc
    emb, tok = mlmc.representation.get("glove50")


    transformed = tok([" ".join(re.split("[/ _-]", x.lower())) for x in classes.keys()])
    mask = (transformed != 0).int()
    l = emb(transformed)
    l = l.sum(-2) / mask.sum(-1, keepdim=True)
    label_embeddings = l/l.norm(p=2,dim=-1, keepdim=True)


    candidates = augment(classes,graph)
    terms = {}
    for i,(k, v) in enumerate(candidates.items()):
        transformed = tok([" ".join(re.split("[/ _-]", x.lower())) for x in v], maxlen=10)
        mask = (transformed != 0).int()
        embeddings = emb(transformed)
        cl = embeddings.sum(-2) / mask.sum(-1, keepdim=True)
        cl = cl / cl.norm(p=2, dim=-1, keepdim=True)
        scores = torch.matmul(label_embeddings[i],cl.t())
        r = scores>0.9
        if r.sum() < 3:
            r[scores.topk(min(3, len(v)))[1]] = True
        if r.sum() >10:
            r=torch.zeros_like(scores)
            topk = scores.topk(10)
            indices = topk[1][topk.values> 0.6]
            r[indices] = True
        terms[k] = [t for t,b in zip(v, r) if b]

    # ToDo: Implement this right !
    result = {}
    for l in  classes.keys():
        children = []
        for u, edge_dict in {x:graph.adj[x] for x in terms[l]}.items():
            if len(edge_dict.keys() )> 0:
                children.extend( list(edge_dict.keys()))

        tmp_graph = nx.subgraph(graph, terms[l]+ children).copy()

        for n in tmp_graph.nodes:
            for v in graph.adj[n].keys():
                if v in tmp_graph.nodes:
                    tmp_graph.add_edge(n,v)
        result[l] = tmp_graph#max(nx.connected_component_subgraphs(tmp_graph.to_undirected()), key=len)
    return result

def embed_align(classes, graph, model="glove50", topk=10, batch_size=50, device="cpu"):
    from mlmc.representation import Embedder
    import re
    e = Embedder(model, device=device, return_device=device)
    classes_tokens = [" ".join(re.split("[/ _-]", x.lower())) for x in classes.keys()]

    class_embeddings = torch.stack([x.mean(-2) for x in e.embed(classes_tokens, None)],0)

    from mlmc.representation.similarities import cos
    scores = []
    for batch in e.embed_batch_iterator(list(graph.nodes), batch_size=batch_size):
        scores.append(cos(class_embeddings, torch.stack([x.mean(-2) for x in batch],0)).t())

    scores = torch.cat(scores, 0)
    similar_nodes = scores.topk(topk, dim=0)[1].t().cpu()
    return {k:[list(graph.nodes)[x.item()] for x in v] for k,v in zip(classes.keys(), similar_nodes)}


def subgraph(classes, graph, augmented, depth=1, allow_non_alignment=False):
    """
    Return a subgraph of relevant nodes.
    Args:
        classes: the classes mapping
        graph:  The graph from which to draw the subgraph from
        depth: The longest path from a classes label to any node

    Returns:
    """
    if not allow_non_alignment:
        assert len(augmented) == len([x for x,v in augmented.items() if len(v) > 0]), \
            "Not every class could be aligned in the graph: "+ ", ".join([x for x,v in augmented.items() if len(v) == 0])
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(classes.keys())
    for key, key_graph in augmented.items():
        subgraph = nx.compose(subgraph,key_graph)
        for n in key_graph.nodes:
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
    print(len(subgraph.edges))
    for node in subgraph.nodes():
        for k, v in graph.adj.get(node,{}).items():
            if k in subgraph.nodes:
                subgraph.add_edge(node, k, label=v["label"])
    print(len(subgraph.edges))
    return subgraph


