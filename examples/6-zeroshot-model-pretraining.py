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
m = mlmc.models.zeroshot.EmbeddingBasedWeighted(
    classes={},
    target="single",
    sformatter=formatter,
    finetune=True,
    mode="max_mean",
    optimizer_params={"lr": 1e-6, "betas": (0.9, 0.99)},
    device="cuda:3",  # If you have a GPU uncomment this
    representation="google/bert_uncased_L-12_H-768_A-12"
)

# This does pretraining on a set of sts data and evluates on agnews
# This should land around 68 % for max_mean model
m.pretrain_sts(datasets=["agnews"])

d = mlmc.data.get("agnews")
m.single()
m.create_labels(d["classes"])
print("After pretraining: ", m.evaluate(d["test"])[1]["accuracy"])