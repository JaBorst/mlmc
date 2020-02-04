import torch
import mlmc
path = "/disk2/jborst/mlmc/embeddings/10K-frequent-words_glove.npz"
import numpy as np
load = np.load(path)
label_embeddings = load["embeddings"]
vocabulary = load["vocabulary"]
vocabulary = dict(zip(vocabulary,range(len(vocabulary))))


weights, vocabulary = mlmc.representation.get_embedding("/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d.txt")

tc2 = mlmc.load(path="/disk2/jborst/mlmc/models/ConceptLSAN_84(post).pt", only_inference=False)
tc2.to(tc2.device)


output, scores, words = tc2(tc2.transform("Hello, this is a text about science and fiction").to(tc2.device), return_scores=True)
[vocabulary[x] for x in concepts.sum(-1).topk(5).indices[0]]