
import torch
import mlmc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = mlmc.data.get_dataset("rcv1", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)
def clean(x):
    import string
    return "".join([c for c in x if c in string.ascii_letters + string.punctuation + " "])
data["train"].transform(clean)
# data["test"].transform(clean)

le = mlmc.graph.get_nmf(data["adjacency"], dim=200)

skg = SKG(data["adjacency"], le, data["classes"], transformer="bert",  #weights=weights, vocabulary=vocabulary,
          optimizer=torch.optim.Adam,
          optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
          loss=torch.nn.BCEWithLogitsLoss,
          device=device)
skg.fit(data["train"], mlmc.data.sample(data["test"],absolute=5000),64, batch_size=16)
skg.evaluate(data["test"])
