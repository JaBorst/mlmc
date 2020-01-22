import mlmc
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
d, classes = mlmc.data.load_appd()
d["adjacency"]

label_embedding = mlmc.graph.get_nmf(d["adjacency"], 35) + 1e-10
from sklearn.manifold import TSNE
#
# X_embedded = TSNE(n_components=2).fit_transform(label_embedding)
# # ax = plt.scatter(X_embedded[:,0], X_embedded[:,1])
# fig, ax = plt.subplots(figsize=(20,20))
# ax.scatter(X_embedded[:,0], X_embedded[:,1])
# for i, txt in enumerate(classes.keys()):
#     ax.annotate(txt, (X_embedded[i,0], X_embedded[i,1]))
# plt.show()

weights, vocabulary = mlmc.helpers.load_static(embedding="/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.100d.txt")

data = mlmc.data.get_dataset("appd", ensure_valid=True, type=mlmc.data.MultiLabelDataset, target_dtype=torch._cast_Float)
tc = mlmc.models.LabelScoringGraphModel(data["classes"], weights, vocabulary,
                                        label_embedding=label_embedding,
                                        adjacency=d["adjacency"],
                                        optimizer=torch.optim.Adam,
                                        optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
                                        loss=torch.nn.BCEWithLogitsLoss,
                                        device=device)
# tc.evaluate(data["valid"])
_ = tc.fit(data["train"], data["valid"], epochs=50, batch_size=32)
tc.evaluate(data["test"], return_report=True)