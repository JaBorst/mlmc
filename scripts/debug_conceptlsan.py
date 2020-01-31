
import torch
import mlmc

epochs = 15
batch_size = 12
mode = "transformer"
transformer = "roberta"
static = None#"/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d_small.txt"
optimizer = torch.optim.Adam
optimizer_params = {"lr": 5e-5, "betas": (0.9, 0.99)}
loss = torch.nn.BCEWithLogitsLoss
dataset = "rcv1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
concept_graph = "wordnet"

import numpy as np
label_embeddings = np.load("/tmp/tmp/mlmc/wordnet_node2vec_100.npz")
label_embeddings = label_embeddings["arr_0"]
label_embeddings = torch.from_numpy(label_embeddings[:20000])


data = mlmc.data.get_dataset(dataset,
                             type=mlmc.data.MultiLabelDataset,
                             ensure_valid=False,
                             valid_split=0.25,
                             target_dtype=torch._cast_Float)
tc = mlmc.models.ConceptLSAN(
    classes=data["classes"],
    label_embed=label_embeddings,
    label_freeze=True,
    use_lstm=False,
    transformer=transformer,
    static=static,
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    loss=loss,
    device=device)

if data["valid"] is None:
    data["valid"] = mlmc.data.sampler(data["test"], absolute=100)

# history=tc.evaluate(data["valid"], batch_size=batch_size,
#                     return_report=True, return_roc=True)
# history=tc.fit(train=mlmc.data.sample(data["train"], absolute=1000), valid= data["valid"], batch_size=batch_size, valid_batch_size=batch_size,epochs=epochs)
history=tc.fit(train=data["train"], valid= data["valid"], batch_size=batch_size, valid_batch_size=batch_size,epochs=epochs)