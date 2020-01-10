import mlmc
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
d, classes = mlmc.data.load_blurbgenrecollection()
d["adjacency"]

classes_rev = {v:k for k,v in classes.items()}
import networkx as nx
from node2vec import Node2Vec

# Create a graph
graph = nx.from_numpy_matrix(d["adjacency"]-np.identity(d["adjacency"].shape[0]))
graph = nx.relabel.relabel_nodes(graph,classes_rev)

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=10, walk_length=30, num_walks=50, workers=4)
model = node2vec.fit(window=50, min_count=1, batch_words=4)

import numpy as np
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2).fit_transform(model.wv.vectors)
# ax = plt.scatter(X_embedded[:,0], X_embedded[:,1])
fig, ax = plt.subplots(figsize=(20,20))
ax.scatter(X_embedded[:,0], X_embedded[:,1])
for i, txt in enumerate(classes.keys()):
    ax.annotate(txt, (X_embedded[i,0], X_embedded[i,1]))
plt.show()



cooc = mlmc.graph.cooc_matrix(d["train"][1], classes)
cooc = cooc + 1000*np.identity(cooc.shape[0])
cooc_rank = np.argsort(cooc,-1)[:,::-1]

embed_rank = np.argsort(np.dot(model.wv.vectors, model.wv.vectors.transpose()),-1,)[:,::-1]

from scipy.stats import spearmanr, kendalltau
spearmans=[]
for (c, ind), a, b in zip(classes.items(),cooc_rank,embed_rank):
    spearmans.append(spearmanr(a,b)[0])
    print(c,spearmans[-1])
print(np.mean(np.abs(spearmans)),np.std(np.abs(spearmans)))
print(np.mean(spearmans),np.std(spearmans))

spearmans=[]
for i in np.where(d["adjacency"].sum(-1) >1)[0]:
    spearmans.append(spearmanr(cooc_rank[i], embed_rank[i])[0])
    print(classes_rev[i], spearmans[-1])
print(np.mean(np.abs(spearmans)),np.std(np.abs(spearmans)))
print(np.mean(spearmans),np.std(spearmans))
