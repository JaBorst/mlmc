
import networkx as nx
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
                        G.add_edge(hh.name(), word.name(), label="is_a")
                        i += 1
        if i >=n:
            break
    return G