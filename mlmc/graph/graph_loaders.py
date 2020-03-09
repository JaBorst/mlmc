
import networkx as nx
import tempfile
from urllib.request import urlopen
import gzip
from ..data.data_loaders import _save_to_tmp, _load_from_tmp



def load_wordnet():
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
                        G.add_edge(hh.name(), word.name(), label= "is_a")

    return G


def load_wordnet_sample(n=1000):
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
                        G.add_edge(hh.name().replace("_"," "), word.name().replace("_"," "), label="is_a")
                        i += 1
        if i >=n:
            break
    return G


def load_NELL():
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