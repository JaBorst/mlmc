from nltk.corpus import wordnet as wn
import networkx as nx

def load_wordnet():
    G = nx.DiGraph()
    for ss in wn.all_synsets():
        if ss.hypernyms() != []:
            for hypernym in ss.hypernyms():
                for hh in hypernym.lemmas():
                    for word in ss.lemmas():
                        G.add_edge(hh.name(), word.name(), label= "is_a")

    return G


def load_wordnet_sample(n=1000):
    G = nx.DiGraph()
    i=0
    for ss in wn.all_synsets():
        if ss.hypernyms() != []:
            for hypernym in ss.hypernyms():
                for hh in hypernym.lemmas():
                    for word in ss.lemmas():
                        G.add_edge(hh.name(), word.name(), label="is_a")
                        i += 1
        if i >=n:
            break
    return G