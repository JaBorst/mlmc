import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
import mlmc


weights, vocabulary = mlmc.helpers.load_static(embedding="/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d_small.txt")
data = mlmc.data.get_dataset("rcv1", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)
tc = mlmc.models.LabelSpecificAttention(data["classes"], weights, vocabulary,
                                        optimizer=torch.optim.Adam,
                                        optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
                                        loss=torch.nn.BCEWithLogitsLoss,
                                        device=device)
tc.evaluate(mlmc.data.sample(data["test"],absolute=10000))

_ = tc.fit(data["train"], mlmc.data.sample(data["test"],absolute=10000), epochs=100, batch_size=32)
tc.evaluate(data["test"])
#

#
#
# tc = mlmc.models.LabelSpecificAttention(data["classes"], weights, vocabulary,dropout=0.5,
#                                     optimizer=torch.optim.Adam,
#                                     optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
#                                     loss=torch.nn.BCEWithLogitsLoss,
#                                     device=device)
#
#
# _ = tc.fit(data["train"], data["test"], epochs=40, batch_size=32)
# tc.evaluate(data["test"])
