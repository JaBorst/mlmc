import numpy as np

def class_random_sampler(idx, labels, k):
    import random
    index = list(range(len(idx)))
    random.shuffle(index)
    sample_x = []
    sample_y = []
    counter = {k: 0 for k in set(labels)}
    for i in index:
        if counter[labels[i]] < k:
            sample_x.append(idx[i])
            sample_y.append(labels[i])
            counter[labels[i]] = counter[labels[i]] + 1
        if all([k <= v for v in counter.values()]):
            break
    return idx

def class_margin_sampler(dataset, k, prior):
    v, i = prior.sort(-1)
    margin = v[:,-1]- v[:,-2]
    _, idx = margin.sort()
    ex = []
    for cls in range(prior.shape[-1]):
        ex.extend(idx[i[:,-1]==cls][:k])
    return type(dataset)(x=[dataset.x[i] for  i in ex], y=[dataset.y[i] for  i in ex], classes=dataset.classes)

def class_certainty_sampler(dataset, k, prior):
    v, i = prior.sort(-1)
    margin = v.max(-1)[1]
    _, idx = margin.sort()
    ex = []
    for cls in range(prior.shape[-1]):
        ex.extend(idx[i[:,-1]==cls][-k:])
    return type(dataset)(x=[dataset.x[i] for  i in ex], y=[dataset.y[i] for  i in ex], classes=dataset.classes)

def class_uncertainty_sampler(dataset, k, prior):
    v, i = prior.sort(-1)
    margin = v.min(-1)[1]
    _, idx = margin.sort()
    ex = []
    for cls in range(prior.shape[-1]):
        ex.extend(idx[i[:,-1]==cls][-k:])
    return type(dataset)(x=[dataset.x[i] for  i in ex], y=[dataset.y[i] for  i in ex], classes=dataset.classes)


def class_medoid_sampler(dataset, k, embedding):
    from sklearn_extra.cluster import KMedoids
    # from sklearn.decomposition import PCA
    # embedding = PCA(n_components=64).fit_transform(embedding)
    cluster = KMedoids(n_clusters=k*len(dataset.classes)).fit(embedding)
    ex = cluster.medoid_indices_
    return type(dataset)(x=[dataset.x[i] for i in ex], y=[dataset.y[i] for i in ex], classes=dataset.classes)

#
# def class_distance_sampler(dataset, k, prior):
#     classes = prior.argmax(-1)
#     for i in range(len(prior))
#     return type(dataset)(x=[dataset.x[i] for i in ex], y=[dataset.y[i] for i in ex], classes=dataset.classes)