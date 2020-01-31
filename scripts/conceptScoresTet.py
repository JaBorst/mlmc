import torch
import mlmc


import numpy as np
label_embeddings = np.load("/tmp/tmp/mlmc/wordnet_node2vec_100.npz")
label_embeddings = label_embeddings["arr_0"]
label_embeddings = torch.from_numpy(label_embeddings)

graph = mlmc.graph.load_wordnet()

tc2 = mlmc.models.load("KimCNN2Branch_rcv1_62.pt")
tc2.to(tc2.device)


output, scores = tc2(tc2.transform("Hello, this is a text about science and fiction").to(tc2.device), return_scores=True)

[list(graph.nodes)[x.item()] for x in torch.sigmoid(scores).topk(3).indices[0]]
