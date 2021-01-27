import mlmc
import torch
from mlmc_lab.mlmc_experimental.models import GR_ranking
from mlmc_lab.mlmc_experimental.loss.LabelwiseRankingLoss import LabelRankingLoss
import mlmc_lab

run=None
percentage=0.0
dataset=""
data = None
graph = "random"
graph_n = 1000
graph_dim = 300
graph_density = 0.2

epochs = 15
batch_size = 50
representation = "google/bert_uncased_L-2_H-768_A-12"  # "distilroberta-base"# #"distilroberta-base"# "google/bert_uncased_L-2_H-768_A-12"#"google/bert_uncased_L-2_H-128_A-2"#"google/bert_uncased_L-4_H-256_A-4"
finetune = True
device = "cuda:1"

optimizer = torch.optim.Adam
optimizer_params = {"lr": 1e-5}
decision_noise = 0.015


zsdata = mlmc.data.get("rcv1")



gr = GR_ranking(classes=zsdata["train"].classes,
        graph_n = graph_n,
        graph_dim = graph_dim,
        graph_density = graph_density,
        loss=LabelRankingLoss(logits=True, add_categorical=2.0, threshold="mcut"),#loss,#torch.nn.BCEWithLogitsLoss if mlmc.data.is_multilabel(zsdata["train"]) else torch.nn.CrossEntropyLoss,
        target="multi" if mlmc.data.is_multilabel(zsdata["train"]) else "single",
        representation = representation,
        finetune=finetune,
        device=device,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        decision_noise=decision_noise)

zsdata["valid"] = mlmc.data.sampler(zsdata["test"], absolute=10000)
d = mlmc_lab.mlmc_experimental.data.ZeroshotDataset(zsdata, zeroshot_classes=mlmc_lab.constants.ZEROSHOT_10["rcv1"])

data =  {"GZSL": d["valid_gzsl"],
                 "ZSL": d["valid_zsl"],
                 "NSL": d["valid_nsl"]}

gr._zeroshot_fit(d)
gr.plot_weights(data)