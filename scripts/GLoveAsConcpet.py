import mlmc
import torch
weights, vocabulary = mlmc.representation.load_static("/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.300d.txt")



epochs = 20
batch_size = 5
mode = "transformer"
representation = "roberta"
optimizer = torch.optim.SGD
optimizer_params = {"lr": 1e-3}#, "betas": (0.9, 0.99)}
loss = torch.nn.BCEWithLogitsLoss
dataset = "blurbgenrecollection"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

sample = mlmc.data.sampler(data["train"],absolute=40)
history=tc.fit(train=sample,
               valid=sample,
               batch_size=batch_size,
               valid_batch_size=batch_size,
               epochs=100)
print("test")
history=tc.fit(train=sample,
               valid=sample,
               batch_size=batch_size,
               valid_batch_size=batch_size,
               epochs=100)
