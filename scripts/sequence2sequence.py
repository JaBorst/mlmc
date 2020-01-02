import sys
sys.path.insert(0,"/tmp/tmp/pycharm_project_765")

import mlmc
import torch
import os
weights, vocabulary = mlmc.helpers.load_glove(embedding="/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d_small.txt")
data = mlmc.data.get_dataset_sequence("conll2003en", sequence_length=140, target_dtype=torch._cast_Long, sparse=True)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


blc = mlmc.models.BILSTMCNN(vocabulary, weights, data["classes"],optimizer=torch.optim.SGD,
                            optimizer_params={"lr": 0.01}, device=device,
                            locked_dropout=0. ,word_dropout=0., dropout=0.3
                            )
# blc = mlmc.models.EmbedderBILSTM(data["classes"],"bert_glove",  optimizer=torch.optim.SGD,
#                             optimizer_params={"lr": 0.01}, device=device,
#                             locked_dropout=0. ,word_dropout=0., dropout=0.25
#                             )

blc.fit(data["train"], data["valid"], epochs=100, batch_size=16)
blc.evaluate(data["test"])

# for b in torch.utils.data.DataLoader(data["train"], batch_size=15): break
# blc(*blc.transform(b["text"], b["labels"]))