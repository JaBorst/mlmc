##########################################################################
#
# Example for  using a zero shot model (experimental feature)
#
##########################################################################

import mlmc, torch

# To improve the out-of-box capabilities of zeroshot-models it is often useful to pretrain. At the moment
# mlmc support two modes of pretraining: nli and sts.
formatter = lambda x: f"The topic of this is {x}"

# Instantiation of the smallest possible model. This should work for any computer.
# m = mlmc.models.zeroshot.EmbeddingBasedWeighted(
#     classes={},
#     target="single",
#     mode="vanilla",
#     finetune=True,
#     optimizer_params={"lr":1e-6},
# )
# If you have a more capable computer and even a GPU, you can use this instantiation to load a larger model
# and put it on the GPU.
m = mlmc.models.zeroshot.EmbeddingBasedWeighted(
    classes={},
    target="single",
    mode="attention_max_mean",
    sformatter=formatter,
    finetune=True,
    optimizer=torch.optim.AdamW,
    optimizer_params={"lr": 1e-6, "betas": (0.9, 0.99)},
    device="cuda:3",  # If you have a GPU uncomment this
    representation="google/bert_uncased_L-12_H-768_A-12"

)

m.pretrain_sts(steps=600, batch_size=12, datasets=["trec6"], eval_every=50)

m.single()
d=mlmc.data.get("trec6")
m.create_labels(d["classes"])
m.evaluate(d["test"])[1]["accuracy"]