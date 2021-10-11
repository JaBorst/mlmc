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
m = mlmc.models.zeroshot.graph.GraphBased(
    classes={},
    target="single",
    sformatter=formatter,
    finetune=True,
    mode="vanilla",
    optimizer_params={"lr": 1e-5, "betas": (0.9, 0.99)},
    optimizer=torch.optim.AdamW,
    device="cuda:3",  # If you have a GPU uncomment this
    representation="google/bert_uncased_L-6_H-768_A-12"
    # representation = "bert-large-uncased",
)

# This does pretraining on a set of sts data and evluates on agnews
# This should land around 68 % for max_mean model
# m.pretrain_mnli(datasets=["trec6","agnews"],batch_size=64, steps=1000, eval_every=50)
from mlmc.data.data_loaders_nli import load_mnli
from mlmc.data.dataset_classes import EntailmentDataset

data, classes = load_mnli()
classes = {'entailment': 2, 'neutral': 1, 'contradiction': 0}
train = EntailmentDataset(x1=data["train_x1"], x2=data["train_x2"], labels=data["train_y"],
                          classes=classes)
test = EntailmentDataset(x1=data["test_x1"], x2=data["test_x2"], labels=data["test_y"],
                         classes=classes)

# history = m.pretrain_entailment(train, valid=test, steps= 5000, eval_every=500, datasets=["trec6","agnews"],batch_size=16)

d = mlmc.data.get("agnews")
m.single()
m.set_sformatter(mlmc.data.SFORMATTER["agnews"])
m.create_labels(d["classes"])
# print("After pretraining: ", m.evaluate(d["test"])[1]["accuracy"])

train = mlmc.data.sampler(d["train"], absolute=40)
print(train.count())


# Use the fit method to train the model
m.single()
# m.loss = mlmc.loss.RelativeRankingLoss()
history = m.fit(train, mlmc.data.sampler(d["test"],absolute=2000), epochs=40)    # RelativeRankingLoss might go to zero. This is not bad thing but you can interrupt
                            # the training at this point or set the number of epochs accordingly



d = mlmc.data.get("dbpedia")
m.single()
m.set_sformatter(mlmc.data.SFORMATTER["dbpedia"])
m.create_labels(d["classes"])
print("After pretraining: ", m.evaluate(mlmc.data.sampler(d["test"],absolute=1000))[1]["accuracy"])

train = mlmc.data.sampler(d["train"], absolute=40)
print(train.count())

d = mlmc.data.get("yahoo_answers")
m.single()
m.set_sformatter(mlmc.data.SFORMATTER["dbpedia"])
m.create_labels(d["classes"])
print("After pretraining: ", m.evaluate(mlmc.data.sampler(d["test"],absolute=1000))[1]["accuracy"])
train = mlmc.data.sampler(d["train"], absolute=400)
print(train.count())



# Use the fit method to train the model
m.single()
# m.loss = mlmc.loss.RelativeRankingLoss()
history = m.fit(train, mlmc.data.sampler(d["test"],absolute=5000), batch_size=16,epochs=50)    # RelativeRankingLoss might go to zero. This is not bad thing but you can interrupt
                            # the training at this point or set the number of epochs accordingly
