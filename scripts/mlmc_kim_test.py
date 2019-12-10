import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
import mlmc


weights, vocabulary = mlmc.helpers.load_glove(embedding="/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d.txt")

data = mlmc.data.get_dataset_only("appd", ensure_valid=True, valid_split=0.25)
tc = mlmc.models.LabelSpecificAttention(data["classes"], weights, vocabulary,
                                        optimizer = torch.optim.Adam,
                                        optimizer_params = {"lr": 0.001},
                                        loss = torch.nn.BCEWithLogitsLoss,
                                        device=device)


_ = tc.fit(data["train"], data["valid"], epochs=100, batch_size=50)
tc.evaluate(data["train"])
