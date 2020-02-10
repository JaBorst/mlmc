import mlmc
import torch
import re
import numpy as np
# weights, vocabulary = mlmc.representation.load_static("/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.300d.txt")
# weights = mlmc.representation.postprocess_embedding(weights)

load = np.load("/tmp/tmp/mlmc/needed_embeddings.npz", allow_pickle=True)
weights = load["weights"]
vocabulary = load["vocabulary"].item()




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


# label_words = [[w if w != "comdedy" else "comedy" for w in re.split("[ /,'’:-]",x.lower())] for x in data["train"].classes.keys()]
# needed_words = list(set([y for x in label_words for y in x if y != ""]))
# import numpy as np
# needed_embeddings = np.zeros((len(needed_words), 300))
# needed_vocabulary = {}
# for i, w in enumerate(needed_words):
#     needed_vocabulary[w]=i
#     needed_embeddings[i]=weights[vocabulary[w]]
# np.savez("needed_embeddings.npz", weights=needed_embeddings, vocabulary=needed_vocabulary, allow_pickle=True)

tc = mlmc.models.GloveConcepts(
    classes=data["classes"],
    concepts=weights,
    label_vocabulary=vocabulary,
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
history=tc.fit(train=train_sample,
               valid=test_sample,
               batch_size=batch_size,
               valid_batch_size=batch_size,
               epochs=2)
i=4
print(test_sample.x[i])
print(test_sample.y[i])
print(tc.additional_concepts(test_sample.x[i], 10))



print("test")
# history=tc.fit(train=train_sample,
#                valid=train_sample,
#                batch_size=batch_size,
#                valid_batch_size=batch_size,
#                epochs=100)
