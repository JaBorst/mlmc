#
# import torch
# import mlmc
#
# epochs = 15
# batch_size = 12
# mode = "transformer"
# transformer = "roberta"
# static = None#"/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d_small.txt"
# optimizer = torch.optim.Adam
# optimizer_params = {"lr": 5e-3, "betas": (0.9, 0.99)}
# loss = torch.nn.BCEWithLogitsLoss
# dataset = "rcv1"
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# concept_graph = "wordnet"
# layers=1
#
# import numpy as np
# label_embeddings = np.load("/tmp/tmp/mlmc/wordnet_node2vec_100.npz")
# label_embeddings = label_embeddings["arr_0"]
# label_embeddings = torch.from_numpy(label_embeddings)[:1000]
#
#
# data = mlmc.data.get_dataset(dataset,
#                              type=mlmc.data.MultiLabelDataset,
#                              ensure_valid=False,
#                              valid_split=0.25,
#                              target_dtype=torch._cast_Float)
# tc = mlmc.models.KimCNN2Branch(
#     classes=data["classes"],
#     representation="roberta",
#     label_embed=label_embeddings,
#     optimizer=optimizer,
#     optimizer_params=optimizer_params,
#     loss=loss,
#     device=device)
#
# if data["valid"] is None:
#     data["valid"] = mlmc.data.sampler(data["test"], absolute=100)
#
# history=tc.evaluate(data["valid"], batch_size=batch_size,
#                     return_report=True, return_roc=True)
# # # history=tc.fit(train=mlmc.data.sample(data["train"], absolute=1000), valid= data["valid"], batch_size=batch_size, valid_batch_size=batch_size,epochs=epochs)
# # history=tc.fit(train=data["train"], valid= data["valid"], batch_size=batch_size, valid_batch_size=batch_size,epochs=epochs)
#
import mlmc
data = mlmc.data.get_multilabel_dataset("blurbgenrecollection")
train_sample = mlmc.data.class_sampler(data["train"], classes=["Business","Philosophy","Marketing"],samples_size=10)
