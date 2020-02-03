path ="/tmp/tmp/mlmc/google-10000-english/google-10000-english-usa.txt"
with open(path) as f: content = [x.replace("\n","") for x in f]

import torch
import mlmc

emb, tok = mlmc.representation.get("/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.200d.txt")

device="cpu"
emb.to(device)
with torch.no_grad(): e = [emb(tok(x, maxlen=10).to(device)).detach().cpu() for x in content ]
e = torch.cat(e)[:,0,:]
import numpy as np
np.savez("/disk2/jborst/mlmc/embeddings/10K-frequent-words_glove.npz", embeddings=e, vocabulary=content)

sim = torch.matmul(e, e.t()) / e.norm(dim=-1)/ e.norm(dim=-1)