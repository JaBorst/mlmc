##########################################################################
#
# Example for  using a zero shot model (highly experimental feature)
#
##########################################################################

import mlmc, torch

# To improve the out-of-box capabilities of zeroshot-models it is often useful to pretrain. At the moment
# mlmc support two modes of pretraining: nli and sts.
formatter = lambda x: f"The topic of this is {x}"


# If you have a more capable computer and even a GPU, you can use this instantiation to load a larger model
# and put it on the GPU.
m = mlmc.models.zeroshot.EmbeddingBasedEntailment(
    classes={},
    target="single",
    sformatter=formatter,
    finetune=False,
    mode="vanilla",
    optimizer_params={"lr": 5e-5, "betas": (0.9, 0.99)},
    device="cuda:0",  # If you have a GPU uncomment this
    representation="google/bert_uncased_L-12_H-768_A-12"
)

# This does pretraining on a set of sts data and evluates on agnews
# This should land around 68 % for max_mean model
# m.pretrain_mnli(datasets=["trec6","agnews"],batch_size=64, steps=1000, eval_every=50)
from mlmc.data.data_loaders_nli import load_mnli
from mlmc.data.datasets import EntailmentDataset

data, classes = load_mnli()
classes = {'entailment': 2, 'neutral': 1, 'contradiction': 0}
train = EntailmentDataset(x1=data["train_x1"], x2=data["train_x2"], labels=data["train_y"],
                          classes=classes)
test = EntailmentDataset(x1=data["test_x1"], x2=data["test_x2"], labels=data["test_y"],
                         classes=classes)
history = m.pretrain_entailment(train, valid=test, steps= 50000, eval_every=1000, datasets=["trec6","agnews"],batch_size=32)
m.embedding.require_grad = True
m.optimizer = torch.optim.AdamW
m.build()
history = m.pretrain_entailment(train, valid=test, steps= 50000, eval_every=1000, datasets=["trec6","agnews"],batch_size=32)

d = mlmc.data.get("agnews")
m.single()
m.set_sformatter(mlmc.data.SFORMATTER["agnews"])
m.create_labels(d["classes"])
print("After pretraining: ", m.evaluate(d["test"])[1]["accuracy"])
