import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
import mlmc


weights, vocabulary = mlmc.helpers.load_glove(embedding="/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.200d.txt")

data = mlmc.data.get_dataset("appd", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)
tc = mlmc.models.KimCNN(data["classes"], weights, vocabulary,
                                        optimizer=torch.optim.Adam,
                                        optimizer_params={"lr": 0.001},
                                        loss=torch.nn.BCEWithLogitsLoss,
                                        device=device)


_ = tc.fit(data["train"], data["test"], epochs=100, batch_size=50)
tc.evaluate(data["test"])
