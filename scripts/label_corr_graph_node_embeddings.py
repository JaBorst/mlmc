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
vectors = mlmc.graph.get_nmf(d["adjacency"],35)
vectors = mlmc.graph.get_node2vec(d["adjacency"],35)

cooc = mlmc.graph.cooc_matrix(d["train"][1], classes)
class_corr = mlmc.graph.helpers.correlate_similarity(cooc, vectors, 10, classwise=True)
for c, co in zip(classes.keys(), class_corr):
    print(c,co)