import torch
import mlmc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = mlmc.data.get_dataset("rcv1", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)

le = mlmc.graph.get_nmf(data["adjacency"],dim=50) + 1e-100
model=mlmc.models.ZAGCNN(
    static="/disk1/users/jborst/Data/Embeddings/fasttext/static/en/wiki-news-300d-10k.vec",
    classes = data["classes"],
    label_embedding=le,
    adjacency=data["adjacency"],
    optimizer=torch.optim.Adam,
    optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
    loss=torch.nn.BCEWithLogitsLoss,
    device="cuda:0")

model.fit(data["train"], mlmc.data.sample(data["test"],absolute=1000), epochs=50,batch_size=50)
model.evaluate(data["test"], return_report=True)