import numpy as np
import itertools

def cooc_matrix(labels, classes):
    coocs = np.zeros((len(classes),len(classes)))
    frequencies = np.zeros((len(classes),1))
    for labelset in labels:
        for p in list(itertools.combinations(labelset, 2)):
            coocs[classes[p[0]],classes[p[1]]] += 1
            coocs[classes[p[1]],classes[p[0]]] += 1
            frequencies[classes[p[1]],0] +=1
            frequencies[classes[p[0]], 0] += 1

    return 1000 * coocs / frequencies / frequencies.transpose()
#
# import mlmc
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# d, classes = mlmc.data.load_blurbgenrecollection()
# cooc = cooc_matrix(d["train"][1], classes)