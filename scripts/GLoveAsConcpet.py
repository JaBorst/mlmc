import mlmc
import torch
weights, vocabulary = mlmc.representation.load_static("/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d.txt")



epochs = 20
batch_size = 32
mode = "transformer"
representation = "roberta"
optimizer = torch.optim.Adam
optimizer_params = {"lr": 5e-3, "betas": (0.9, 0.99)}
loss = torch.nn.BCEWithLogitsLoss
dataset = "movies_summaries"
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
concept_graph = "random"
layers = 1
label_freeze = True
description= "LSAN extension with Glove embeddings."




data = mlmc.data.get_dataset(dataset,
                             type=mlmc.data.MultiLabelDataset,
                             ensure_valid=False,
                             valid_split=0.25,
                             target_dtype=torch._cast_Float)
tc = mlmc.models.GloveConcepts(
    classes=data["classes"],
    concepts=weights,
    label_vocabulary=vocabulary,
    label_freeze=label_freeze,
    representation=representation,
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    loss=loss,
    device=device)

if data["valid"] is None:
    data["valid"] = mlmc.data.sampler(data["test"], absolute=5000)

history=tc.fit(train=data["train"],
               valid=data["valid"],
               batch_size=batch_size,
               valid_batch_size=batch_size,
               epochs=epochs)
