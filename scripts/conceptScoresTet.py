import torch
import mlmc

mode = "transformer"
transformer = "roberta"
static = None  # "/disk1/users/jborst/Data/Emb


loaded = torch.load("ConceptGraph_Wordnet_cnn_69.pt")
epochs = 35
batch_size = 12
mode = "transformer"
transformer = None#"roberta"
static = "/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d_small.txt"
optimizer = torch.optim.Adam
optimizer_params = {"lr": 5e-3, "betas": (0.9, 0.99)}
loss = torch.nn.BCEWithLogitsLoss
dataset = "rcv1"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
concept_graph = "wordnet"
layers = 1
label_freeze = True
description = "2Branch KimCNN Test when training with Wordnet."
import numpy as np
label_embeddings = np.load("/tmp/tmp/mlmc/wordnet_node2vec_100.npz")
label_embeddings = label_embeddings["arr_0"]
label_embeddings = torch.from_numpy(label_embeddings)

data = mlmc.data.get_dataset(dataset,
                             type=mlmc.data.MultiLabelDataset,
                             ensure_valid=False,
                             valid_split=0.25,
                             target_dtype=torch._cast_Float)
tc = mlmc.models.KimCNN2Branch(
    classes=data["classes"],
    label_embed=label_embeddings,
    label_freeze=label_freeze,
    use_lstm=False,
    transformer=transformer,
    static=static,
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    loss=loss,
    device=device)
tc.load_state_dict(loaded["model_state_dict"])

