
import mlmc
import torch
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

with open("/disk1/users/jborst/Data/Test/MultiLabel/reuters/corpus-reuters-corpus-vol1/topic_codes.txt","r") as f:
    topics=[x.replace("\n","").split("\t") for x in f.readlines() if len(x) > 1][2:]
topicmap={x[0]:x[1] for x in topics}

weights, vocabulary = mlmc.helpers.load_glove(embedding="/disk1/users/jborst/Data/Embeddings/fasttext/static/en/wiki-news-300d-10k.vec")
data = mlmc.data.get_dataset("blurbgenrecollection", type=mlmc.data.MultiLabelDataset, ensure_valid=False, valid_split=0.25, target_dtype=torch._cast_Float)

# label_embed = np.stack([np.https://www.aclweb.org/anthology/D19-1426.pdfmean([weights[vocabulary.get(y.lower(),0)] for y in topicmap[x].split(" ") ],0) for x in data["classes"].keys() if len(x)>1])
tc = mlmc.models.LSANOriginal(lstm_hid_dim=300,
                                         d_a=200,
                                         label_embed=torch.rand((len(data["classes"]),300)),#label_embed,
                                         weights=weights,
                                         max_len=400,
                                         classes = data["classes"],
                                         vocabulary=vocabulary,
                                        optimizer=torch.optim.Adam,
                                        optimizer_params={"lr": 0.001, "betas": (0.9, 0.99)},
                                        loss=torch.nn.BCEWithLogitsLoss,
                                        device=device)
# tc.evaluate(mlmc.data.sample(data["test"],absolute=10000))

_ = tc.fit(data["train"], mlmc.data.sample(data["test"],absolute=10000), epochs=50, batch_size=32)
tc.evaluate(data["test"], return_report=True)
