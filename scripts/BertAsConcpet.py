import mlmc
import torch
import re
import numpy as np



epochs = 20
batch_size = 32
mode = "transformer"
representation = "roberta"
optimizer = torch.optim.Adam
optimizer_params = {"lr": 1e-4}#, "betas": (0.9, 0.99)}
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



tc = mlmc.models.BertAsConcept(
    classes=data["classes"],
    label_freeze=label_freeze,
    representation=representation,
    optimizer=optimizer,
    # optimizer_params=optimizer_params,
    loss=loss,
    device=device)

if data["valid"] is None:
    data["valid"] = mlmc.data.sampler(data["test"], absolute=50)

train_sample = mlmc.data.class_sampler(data["train"], classes=["Business"],samples_size=100)
test_sample = mlmc.data.class_sampler(data["train"], classes=["Business"],samples_size=100)
history=tc.fit(train=mlmc.data.sampler(data["train"], absolute=10000),
               valid=mlmc.data.sampler(data["valid"], absolute=10000),
               batch_size=batch_size,
               valid_batch_size=batch_size,
               epochs=50)
i=4
print(test_sample.x[i])
print(test_sample.y[i])
print(tc.additional_concepts(test_sample.x[i], 10))



print("test")
history=tc.fit(train=train_sample,
               valid=train_sample,
               batch_size=batch_size,
               valid_batch_size=batch_size,
               epochs=100)
