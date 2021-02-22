# Find edges to insert other nodes
import re
from collections import Counter

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

def ngrams(x, k):
    """
    Splits text into n-grams.

    :param x: A string
    :param k: Size of each n-gram
    :return: List of n-grams
    """
    return [x[i:(i + k)] for i in range(len(x) - k + 1)]

def _mh(x, k, num_perm, wk):
    """
    Calculates MinHash of a string for estimating Jaccard similarity using shingling. Multiple shingling sizes may be
    specified.

    :param x: A string
    :param k: Shingling size(s)
    :param num_perm: Number of permutation functions
    :return: MinHash
    """
    from datasketch import MinHash, MinHashLSH
    x = x.upper()
    k = k if isinstance(k, (tuple, list)) else [k]
    m1 = MinHash(num_perm)
    for kk in k:
        for i in ngrams(x, kk): m1.update(i.encode("utf8"))
    for w in x.split(" "):
        m1.update(w.encode("utf8"))
    #     for kk in k:
    #         for i in ngrams(w, kk): m1.update(i.encode("utf8"))
    return m1

def _subwordsplits_mh(x, k,num_perm, wk):
    """
    Calculates MinHash of a string for estimating Jaccard similarity using shingling. The initial shingling is further
    split in substrings. Multiple shingling sizes may be specified.

    :param x: A string
    :param k: Shingling size(s)
    :param num_perm: Number of permutation functions
    :param wk: Shingling size(s) of the substrings
    :return: MinHash
    """
    x = x.upper()
    from datasketch import MinHash, MinHashLSH
    k = k if isinstance(k, (tuple, list)) else [k]
    wk = wk if isinstance(wk, (tuple, list)) else [wk]
    m1 = MinHash(num_perm)
    for wkk in wk:
        for w in ngrams(x.split(" "), wkk):
            w = " ".join(w)
            for kk in k:
                for i in ngrams(w, kk): m1.update(i.encode("utf8"))
    return m1

def edges(l1, l2,_mh, num_perm=48, n=(2, 3), threshold=0.65, wk=(1,2,3)):
    """
    Compares two lists using Jaccard similarity and shingling. Multiple shingling sizes may be specified.

    :param l1: List of nodes
    :param l2: List of nodes
    :param _mh: MinHash function (_mh or _subwordsplits_mh)
    :param num_perm: Number of permutation functions
    :param n: Shingling size(s)
    :param threshold: Jaccard similarity threshold
    :param wk: Shingling size(s) of the substrings
    :return: Dictionary containing all objects of l1 as keys and the corresponding objects of l2 above the specified
    threshold as values in list form.
    """
    # helper for
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for x in l2:
        lsh.insert(x, _mh(x, n, num_perm=num_perm, wk = wk))
    sims = {k:lsh.query(_mh(k, n,  num_perm=num_perm, wk = wk)) for k in l1}

    return sims



def graph_insert_labels(data, kb, explanations):
    # TODO: Documentation

    l1 = list(data["classes"].keys())
    l2 = list(kb.nodes())

    # find the nodes that are already in graph
    clean_dict = {x:x.replace("/"," ").replace("Ec ", "EU ").replace("ec ","EU ").replace(", "," ").replace(":"," ").replace("’"," ").replace("&"," ") for x in l1}

    nodes_list = {}
    for n in kb:
        for l in l1:
            if clean_dict[l].upper() == n.upper() or (clean_dict[l].endswith("s") and (clean_dict[l][:-1].lower() == n.lower() )):
                if clean_dict[l] not in nodes_list: nodes_list[l] = []
                nodes_list[l].append(n)
    candidates = {l:[x for x in kb if any(w.upper() in x.upper() for w in l.split())] for l in clean_dict.values() if l not in nodes_list }
    nodes_list.update(edges([x for x in clean_dict.values() if x not in nodes_list.keys()], list(set(sum(candidates.values(),[]))),num_perm=72, n=(2,3,4), threshold=0.9, wk=(1,),_mh=_subwordsplits_mh))
    nodes_list = { k:v for k,v in nodes_list.items() if v != []}


    edges_list = {l:[x for x in  kb for w in [" ".join(x) for i in [1,2,3]  for x in ngrams(list(re.split("[ /,-]+",clean_dict[l])),i)] if len(x) >3 and (w.lower() == x.lower()  or (w.endswith("s") and w[:-1].lower() == x.lower()) )] for l in clean_dict.keys() if l not in nodes_list.keys()}
    edges_list = { k:v for k,v in edges_list.items() if v != []}
    from datasketch import MinHash, MinHashLSH

    from stanza import Pipeline
    stanza = Pipeline(lang="en", processors='tokenize,pos', use_gpu=True)
    tokenized_explanations = {k: k.split() + [x["text"] for x in stanza(v).to_dict()[0] if x["upos"]=="NOUN" or x["upos"]=="VERB" or x["upos"]=="ADJ" ] for k,v in explanations.items()}
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([" ".join(x).lower() for k,x in tokenized_explanations.items()]+ list([x.lower() for x in kb]))

    threshold = 0.2
    classes_X = X[:len(data["classes"])]>threshold
    nodes_X = X[len(data["classes"]):]>threshold

    r = classes_X.__mul__(nodes_X.transpose())
    ex_edges = [(list(tokenized_explanations.keys())[i],l2[j]) for i,j in zip(*(r>0).nonzero())]
    ex_edges = {cls: [x[1] for x in ex_edges if x[0] == cls]  for cls in explanations.keys()}

    paths2 = [[ (k,n1,n2) for n1 in v for n2 in v if nx.has_path(kb, n1, n2)] for k,v in ex_edges.items()]
    main_nodes = {x[0][0]:[k for k, v in Counter([y[1] for y in x]).items() if v > 5] for x in paths2}
    intra_edges = [(n1,n2) for n1 in data["classes"] for n2 in data["classes"] if n1 != n2 and any([x==x2  for x in clean_dict[n1].split() for x2 in clean_dict[n2].split()])]

    label_subgraph = nx.OrderedDiGraph()
    label_subgraph.add_nodes_from(data["classes"].keys())
    label_subgraph.add_nodes_from(kb.nodes)
    label_subgraph.add_edges_from(kb.edges)

    label_subgraph = nx.relabel_nodes(label_subgraph, {v[0]:k for k,v in nodes_list.items()})

    label_subgraph.add_edges_from([(n,k) for k, v in edges_list.items() for n in v])
    label_subgraph.add_edges_from([(n,k) for k, v in main_nodes.items() for n in v])
    label_subgraph.add_edges_from(intra_edges)

    assert all([x in list(label_subgraph.nodes)[:len(data["classes"])] for x in data["classes"].keys()]), "Error lost classes underway"
    [(k,label_subgraph.in_degree(k), label_subgraph.out_degree(k), label_subgraph.in_edges(k) if label_subgraph.in_degree(k) <3 else [])for k,x in data["classes"].items()]#data["classes"] ]

    return label_subgraph

