import numpy as np
import itertools
from sklearn.manifold import TSNE


def cooc_matrix(labels, classes):
    """Deprecated"""
    coocs = np.zeros((len(classes),len(classes)))
    frequencies = np.zeros((len(classes),1))
    for labelset in labels:
        for p in list(itertools.combinations(labelset, 2)):
            coocs[classes[p[0]],classes[p[1]]] += 1
            coocs[classes[p[1]],classes[p[0]]] += 1
            frequencies[classes[p[1]],0] +=1
            frequencies[classes[p[0]], 0] += 1

    return 1000 * coocs / frequencies / frequencies.transpose()


def correlate_similarity(coocs, embeddings, n, classwise=False, corr="spearman"):
    """Deprecated"""
    cooc_rank = np.argsort(coocs, -1)[:, -n::-1]
    embed_rank = np.argsort(np.dot(embeddings, embeddings.transpose()), -1, )[:, -n::-1]

    from scipy.stats import spearmanr, kendalltau
    classcorrelations = []
    if corr=="spearman": fct = spearmanr
    if corr=="kendalltau": fct = kendalltau
    for  a, b in zip(cooc_rank, embed_rank):
        classcorrelations.append(fct(a, b)[0])

    if classwise:
        return classcorrelations
    else:
        return [np.mean(np.abs(classcorrelations)), np.std(np.abs(classcorrelations))],\
               [np.mean(classcorrelations), np.std(classcorrelations)]

def show_graph(vectors, classes):
    """
    Plot A TSNE
    :param vectors: Vectors to plot
    :param classes: Labels of the vectors to plot
    :return:
    """
    import matplotlib.pyplot as plt
    X_embedded = TSNE(n_components=2).fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1])
    for i, txt in enumerate(classes.keys()):
        ax.annotate(txt, (X_embedded[i, 0], X_embedded[i, 1]))
    plt.show()
