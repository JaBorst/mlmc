import gzip
import tempfile
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import networkx as nx
import rdflib
from rdflib import Graph as RDFGraph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
from tqdm import tqdm

from ..data.data_loaders import _save_to_tmp, _load_from_tmp


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

def load_wordnet():
    """
    Loading the wordnet graph as a networkx.DiGraph
    :return: A networkx.DiGraph containing wordnet.
    """
    try:
        from nltk.corpus import wordnet as wn
    except:
        print("To use this function you have to install nltk.")
    G = nx.DiGraph()
    for ss in wn.all_synsets():
        if ss.hypernyms() != []:
            for hypernym in ss.hypernyms():
                for hh in hypernym.lemmas():
                    for word in ss.lemmas():
                        G.add_edge(hh.name().replace("_"," "), word.name().replace("_"," "), label="related")

    return G


def load_wordnet_sample(num=1000):
    """
    Loading the wordnet graph as a networkx.DiGraph
    :num: The size of the subsample ( Only the first n synsets will be added )
    :return: A networkx.DiGraph containing wordnet.
    """
    try:
        from nltk.corpus import wordnet as wn
    except:
        print("To use this function you have to install nltk.")
    G = nx.DiGraph()
    i=0
    for ss in wn.all_synsets():
        if ss.hypernyms() != []:
            for hypernym in ss.hypernyms():
                for hh in hypernym.lemmas():
                    for word in ss.lemmas():
                        G.add_edge(hh.name().replace("_"," "), word.name().replace("_"," "), label="related")
                        i += 1
        if i >=num:
            break
    return G


def load_NELL():
    """
    Download and return the NELL Graph (Never Ending Lanuage Learning
    Will return only elements that have a confidence of 0.999 and larger.
    :return: A networkx.DiGraph
    """
    url = "http://rtw.ml.cmu.edu/resources/results/08m/NELL.08m.1115.esv.csv.gz"
    data = _load_from_tmp("NELL")
    if data is not None:
        return data
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            resp = urlopen(url)
            tf = gzip.open(resp, mode="r")
            content = tf.read()
            x = content.decode().split("\n")

            import networkx as nx
            G = nx.DiGraph()
            for line in x[1:]:
                test = line.split("\t")
                if len(test) < 10: continue
                confidence = test[4]
                if float(confidence) > 0.999:
                    relation = test[1]
                    if relation != "generalizations" and relation != "concept:latitudelongitude":
                        entityLiterals = test[8].lower()
                        valueLiterals = test[9].lower()
                        G.add_edge(entityLiterals.replace("_"," "), valueLiterals.replace("_"," "), label=relation, confidence=confidence)
                    if relation == "generalizations":
                        entityLiterals = test[8].lower()
                        valueLiterals = test[2].lower().split(":")[-1]
                        G.add_edge(entityLiterals.replace("_"," "), valueLiterals.replace("_"," "), label=relation, confidence=confidence)
        _save_to_tmp("NELL",G)
        return G
