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
    """
    Transforms rdflib terms into plain text.

    :param x: A term (see rdflib.term)
    :param rg: An RDF Graph
    :param lang: Language of the wikidata article
    :return: String representation of term
    """
    if isinstance(x, rdflib.term.URIRef):
        from rdflib.namespace import SKOS
        if str(SKOS) in str(x): return [x.split("#")[-1]]
        # elif "zbw.eu" in str(x): return ["stw_"+ x.split("/")[-1]]
        elif "wikidata" in str(x): return[str(x)]
        else:
            return ([str(l[1]) for l in rg.preferredLabel(x) if l[1].language==lang], "stw_"+ x.split("/")[-1])
    elif isinstance(x, rdflib.term.Literal):
        if x.language != lang: return None
        else: return [str(x)]
    elif isinstance(x, rdflib.term.BNode):
        return None#[str(x)]
    return (x)


def transform_triples(rg,lang="en"):
    """
    Transforms rdflib triples into plain text.

    :param rg: An RDF Graph
    :param lang: Language of the wikidata article
    :return: List containing transformed triples
    """
    new_list =  [[transform(x, rg, lang) for x in t] for t in tqdm(rg)]
    new_list = [x for x in new_list
                if (not (any([r==[] for r in x]) or any([r is None for r in x]))) and ( "prefLabel" not in x[1])]
    return new_list

def get_wikidata_desc(x):
    """
    Queries wikidata using ID's to retrieve descriptions.

    :param x: List of ID's
    :return: Dictionary of input ID's corresponding to its wikidata descriptions
    """
    import requests
    output = []

    batch_size = 150
    for ind in tqdm(list(range(0,len(x),batch_size))):
        batch = x[ind:(ind+batch_size)]
        clause = '||'.join(['?att="'+i+'"' for i in batch if len(i) < 10])

        query = f"SELECT ?item ?itemLabel ?att ?itemDescription" \
                f"        WHERE{{?item wdt:P3911 ?att." \
                f"              Filter ({clause}). " \
                f"SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\".}}}}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

        r = requests.get('https://query.wikidata.org/sparql', params={'format': 'json', 'query': query},headers=headers)
        data = r.json()

        for item in data['results']['bindings']:
            output.append(dict({
                label: item[label]['value'] for label in item.keys() if label in item}))
    descriptions = {d["att"]: d["itemLabel"] + " is a " + d["itemDescription"] for d in output if "itemDescription" in d.keys()}
    return descriptions

def load_mesh():
    """
    Loading the MeSH thesaurus graph as a networkx.DiGraph.

    :return: A networkx.DiGraph containing MeSH.
    """

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
    """
    Loading the NASA thesaurus graph as a networkx.DiGraph.

    :return: A networkx.DiGraph containing the NASA Thesaurus.
    """
    data = _load_from_tmp("nasa")
    if data is not None:
        return data
    else:
        url = "https://www.sti.nasa.gov/download/NASA_Thesaurus_Alpha_SKOS.xml" #TODO: Fix broken link
        response = urlopen(url)
        content = response.read().decode("utf-16")
        rg = RDFGraph()
        rg = rg.parse(data=content)
        literal_triples = transform_triples(rg)
        g = nx.OrderedDiGraph()
        for triple in literal_triples:
            for u in triple[0]:
                for v in triple[2]:
                    for p in triple[1]:
                        g.add_edge(u, v, label=p)
        _save_to_tmp("nasa", g)
        return g



def load_gesis():
    """
    Loading the gesis TheSoz graph as a networkx.DiGraph.

    :return: A networkx.DiGraph containing TheSoz.
    """
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
        g = nx.OrderedDiGraph()
        for triple in literal_triples:
            for u in triple[0]:
                for v in triple[2]:
                    for p in triple[1]:
                        g.add_edge(u, v, label = p)
        _save_to_tmp("gesis",g)
        return g

def load_stw():
    """
    Loading the STW Thesaurus for Economics graph as a networkx.DiGraph.

    :return: A networkx.DiGraph containing the STW Thesaurus for Economics.
    """
    data = _load_from_tmp("stw")
    if data is not None:
        return data
    else:
        url = "https://aspra29.informatik.uni-leipzig.de/stw.rdf.zip"
        response = urlopen(url)
        zf = ZipFile(BytesIO(response.read()))
        with zf.open("stw.rdf") as f:
            content = f.read()
        rg = RDFGraph()
        rg = rg.parse(data=content)
        stw_ids = [str(x).split("/")[-1] for x in rg.all_nodes() if "zb" in str(x)]
        descriptions = get_wikidata_desc(stw_ids)
        literal_triples = transform_triples(rg)
        g = nx.OrderedDiGraph()
        for triple in tqdm(literal_triples):
            u,p,v = triple
            if isinstance(p, list):
                pr = p[0]
            if isinstance(p, tuple):
                pr = p[1]
            if pr == "inScheme" or pr == "historyNote" or u == [] or v ==[] or p ==[] or p[0]==[]:
                continue
            else:
                if u[0] == [] or v[0] == []: continue
                if isinstance(u,tuple):
                    if u[1].split("_")[-1] in descriptions:
                        g.add_node(u[0][0], stw=u[1].split("_")[-1], description=descriptions[u[1].split("_")[-1]])
                    else:
                        g.add_node(u[0][0], stw=u[1].split("_")[-1])
                    m=u[0]
                else:
                    m=u
                if isinstance(v, tuple):
                    if v[1].split("_")[-1] in descriptions:
                        g.add_node(v[0][0], stw=v[1].split("_")[-1], description=descriptions[v[1].split("_")[-1]])
                    else:
                        g.add_node(v[0][0], stw=v[1].split("_")[-1])
                    n=v[0]
                else:
                    n=v
                g.add_edge(m[0], n[0], label=pr[0])
        url2 ="https://aspra29.informatik.uni-leipzig.de/stw_wikidata_mapping.rdf.zip"
        response = urlopen(url2)
        zf = ZipFile(BytesIO(response.read()))
        with zf.open("stw_wikidata_mapping.rdf") as f:
            content = f.read()
        rg2 = RDFGraph()
        rg2 = rg2.parse(data=content)
        mapping = transform_triples(rg2, "en")
        mapping = [(str(x[0][0]),str(x[2][0][0])) if "wikidata.org" in str(x[0][0]) else (str(x[2][0]),str(x[0][0][0])) for x in mapping if "Match" in x[1][0]]

        for wid, node in mapping:
            g.node[node]["wikidata"] = wid

        from .graph_operations import augment_wikiabstracts
        g = augment_wikiabstracts(g)
        try:
            g.remove_node("STW Thesaurus for Economics")
        except:
            pass
        _save_to_tmp("stw",g)
        return g

def load_elsevier():
    """
    Loading the Elsevier graph used for Elsevier Fingerprint Engine as a networkx.DiGraph

    :return: A networkx.DiGraph containing Elsevier.
    """
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
    data = _load_from_tmp("wordnet")
    if data is not None:
        return data
    else:
        try:
            from nltk.corpus import wordnet as wn
            import nltk
            nltk.download('wordnet')
        except:
            print("To use this function you have to install nltk.")
        G = nx.OrderedDiGraph()
        for ss in wn.all_synsets():
            if ss.hypernyms() != []:
                for hypernym in ss.hypernyms():
                    for hh in hypernym.lemmas():
                        for word in ss.lemmas():
                            G.add_node(hh.name().replace("_"," "), label=hh.name().replace("_"," "))
                            G.add_node(word.name().replace("_"," "), label=word.name().replace("_"," "))
                            G.add_edge(hh.name().replace("_"," "), word.name().replace("_"," "), label="hypernym")
                            for word2 in ss.lemmas():
                                G.add_edge(word2.name().replace("_"," "), word.name().replace("_"," "), label="synonym")
            if ss.hyponyms() != []:
                for hyponym in ss.hyponyms():
                    for hh in hyponym.lemmas():
                        for word in ss.lemmas():
                            G.add_node(hh.name().replace("_"," "), label=hh.name().replace("_"," "))
                            G.add_node(word.name().replace("_"," "), label=word.name().replace("_"," "))
                            G.add_edge(hh.name().replace("_"," "), word.name().replace("_"," "), label="hyponym")

        _save_to_tmp("wordnet",G)
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
    G = nx.OrderedDiGraph()
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
            G = nx.OrderedDiGraph()
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



def load_conceptNet():
    """
    Loading the conceptNet graph as a networkx.DiGraph.

    :return: A networkx.DiGraph containing conceptNet.
    """
    url = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
    data = _load_from_tmp("conceptnet")
    if data is not None:
        return data
    else:
        resp = urlopen(url)
        tf = gzip.open(resp, mode="r")
        g = nx.DiGraph()
        relations=[]
        for line in tqdm(tf):
            line = line.decode("utf8")#.split("\t")
            triple = [[y for y in x.split("/") if y != ""] for x in line.split("\t")[0].split("[")[1].split(",/")]
            relations.append(triple[0][1])
            if triple[1][1] == "en" and triple[2][1] == "en" and any([x in triple[0] for x in ["IsA", "HasProperty", "Synonym", "UsedFor", "PartOf", "RelatedTo"]]):
                triple = [x[1] if x[0] == "r" else x[2] for x in triple[:3]]
                g.add_node(triple[1])
                g.add_node(triple[2])
                g.add_edge(triple[1], triple[2], label=triple[0])
    _save_to_tmp("conceptnet",g)
    return g

