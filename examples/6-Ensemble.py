##########################################################################
#
# Example for  using a zero shot model (highly experimental feature)
#
##########################################################################

import mlmc, torch

# To improve the out-of-box capabilities of zeroshot-models it is often useful to pretrain. At the moment
# mlmc support two modes of pretraining: nli and sts.
formatter = lambda x: f"The topic of this is {x}"




data, classes = mlmc.data.load_agnews()
data = {k: mlmc.data.SingleLabelDataset(x=v[0], y=v[1], classes=classes) for k,v in data.items()}
data = {k: mlmc.data.sampler(v, absolute=100) for k,v in data.items()}

# If you have a more capable computer and even a GPU, you can use this instantiation to load a larger model
# and put it on the GPU.
m = [
    mlmc.models.zeroshot.Siamese(classes=classes, finetune="all",sformatter=formatter, representation="sentence-transformers/all-mpnet-base-v2"),
    mlmc.models.zeroshot.Siamese(classes=classes, finetune="all", sformatter=formatter, representation="sentence-transformers/multi-qa-distilbert-cos-v1"),
    mlmc.models.zeroshot.Siamese(classes=classes, finetune="all", sformatter=formatter, representation="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
    mlmc.models.zeroshot.Encoder(classes=classes, finetune="all", sformatter=formatter, representation="textattack/bert-base-uncased-MNLI")
]

ensemble = mlmc.ensembles.Ensemble(m)

ensemble.evaluate_ensemble(data["test"], batch_size=10)
ensemble.predict_ensemble(data["test"].x, vote=False)