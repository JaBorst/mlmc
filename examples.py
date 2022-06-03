from mlmc.data import Augmenter, Corpus, sampler
from mlmc.data.sampler import get
from mlmc.models import Siamese
import mlmc_lab.mlmc_experimental as mlmce

data = mlmce.data.get("agnews")


pretrain = sampler(mlmce.data.get("dbpedia")["train"], absolute=100)
a = Augmenter("sometimes", wordnet=0.1)
m = lambda : Siamese(classes=data["classes"], target="single", finetune="all", word_cutoff=0.1, word_noise=0.1, feature_cutoff=0.1)
k = 100

from mlmc.pipelines import TransferTest


t = TransferTest(model=m, strategy="random",  augmenter=a)
t.pretrain(train=pretrain)
t.set_dataset(data["train"], data["test"], data["test"])
t.active_transfer(k=100, epochs=100)