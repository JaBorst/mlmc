import mlmc
import torch
import re
import numpy as np
from apex import amp



epochs = 30
batch_size = 24
mode = "transformer"
representation = "roberta"
optimizer = torch.optim.Adam
optimizer_params = {"lr": 1e-6}#, "betas": (0.9, 0.99)}
loss = torch.nn.BCEWithLogitsLoss
dataset = "blurbgenrecollection"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
concept_graph = "random"
layers = 1
label_freeze = True
description= "LSAN extension with Glove embeddings."




data = mlmc.data.get_dataset(dataset,
                             type=mlmc.data.MultiLabelDataset,
                             ensure_valid=False,
                             valid_split=0.25,
                             target_dtype=torch._cast_Float)

# data2 = mlmc.data.get_dataset("rcv1",
#                              type=mlmc.data.MultiLabelDataset,
#                              ensure_valid=False,
#                              valid_split=0.25,
#                              target_dtype=torch._cast_Float)
#
# # CHange topic descriptions
# data2["classes"]={data2["topicmap"][k].capitalize():v for k,v in data2["classes"].items()}
# for key in ("train","test"):
#     data2[key].y = [[data2["topicmap"][l].capitalize() for l in labellist]for labellist in data2[key].y]
#     data2[key].classes = data2["classes"]


tc = mlmc.models.BertAsConceptFineTuning(
    classes=data["classes"],
    label_freeze=label_freeze,
    representation=representation,
    optimizer=optimizer,
    optimizer_params=optimizer_params,
    loss=loss,
    max_len=200,
    device=device)

tc, optimizer = amp.initialize(tc, tc.optimizer, opt_level="O1")
# tc.embedding.half()

if data["valid"] is None:
    data["valid"] = mlmc.data.sampler(data["test"], absolute=50)

train_sample = mlmc.data.class_sampler(data["train"], classes=["Business"],samples_size=100)
test_sample = mlmc.data.class_sampler(data["train"], classes=["Business"],samples_size=100)
history=tc.fit(train=data["train"],
               valid=data["valid"],
               batch_size=32,
               valid_batch_size=batch_size,
               epochs=10)

##################
#
#  SWITCH LABELS
#

if data2["valid"] is None:
    data2["valid"] = mlmc.data.sampler(data2["test"], absolute=3000)

tc.create_labels(data2["classes"])
print(tc.evaluate(data2["valid"]))
history=tc.fit(train=data2["train"],
               valid=data2["valid"],
               batch_size=batch_size,
               valid_batch_size=batch_size,
               epochs=1)

mlmc.save(tc,"bert_as_concept_0.pt", only_inference=False)
i=4
# print(test_sample.x[i])
# print(test_sample.y[i])
# print(tc.additional_concepts(test_sample.x[i], 10))
#


print("test")
history=tc.fit(train=train_sample,
               valid=train_sample,
               batch_size=batch_size,
               valid_batch_size=batch_size,
               epochs=100)
