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
node2vec = Node2Vec(graph, dimensions=10, walk_length=30, num_walks=500, workers=4)
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

weights, vocabulary = mlmc.helpers.load_glove(embedding="/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d.txt")

data = mlmc.data.get_dataset("blurbgenrecollection", type=mlmc.data.MultiLabelDataset, target_dtype=torch._cast_Float)
tc = mlmc.models.LabelScoringModel(data["classes"], weights, vocabulary,
                                   label_embedding=model.wv.vectors,
                                        optimizer=torch.optim.Adam,
                                        optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
                                        loss=torch.nn.BCEWithLogitsLoss,
                                        device=device)
# tc.evaluate(data["valid"])
_ = tc.fit(data["train"], data["valid"], epochs=20, batch_size=32)
tc.evaluate(data["test"], return_report=True)