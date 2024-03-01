##########################################################################
#
# Example for  using a zero shot model (highly experimental feature)
#
##########################################################################

import mlmc
from mlmc.models import BayesNetwork
# To improve the out-of-box capabilities of zeroshot-models it is often useful to pretrain. At the moment
# mlmc support two modes of pretraining: nli and sts.
formatter = lambda x: f"The topic of this is {x}"


# If you have a more capable computer and even a GPU, you can use this instantiation to load a larger model
# and put it on the GPU.
m = mlmc.models.zeroshot.Encoder(
    classes={},
    target="single",
    representation="textattack/bert-base-uncased-MNLI",
    device="cuda:0"
)


data, classes = mlmc.data.load_agnews()
data = {k: mlmc.data.SingleLabelDataset(x=v[0], y=v[1], classes=classes) for k,v in data.items()}
data = {k: mlmc.data.sampler(v, absolute=100) for k,v in data.items()}

m.create_labels(classes)

m = BayesNetwork(m)


labels, scores, variance, prediction = m.bayesian_predict_batch(data["test"].x, batchsize=10, return_scores=True, p=0.1)
print("\n".join([f"{label} {score}\n {text}\n" for text, label, score in zip(data["test"].x, labels, prediction.max(-1)[0].tolist())]))

