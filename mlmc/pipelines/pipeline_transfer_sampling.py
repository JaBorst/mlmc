from mlmc.data.sampler import get
from mlmc.data import validation_split

class Transfer:
    def __init__(self, model, strategy, valid_split=0.2, log_mlflow=False, target="single"):
        assert callable(model), "model should be a callable creating the model"
        self._create_model = model
        self.strategy = get(strategy)
        self._prior=None
        self._embedding=None
        self._pretrain=False
        self._log_mlflow = log_mlflow
        self._valid_split = valid_split
        self._target = target

    def set_dataset(self, corpus, classes):
        self.corpus = corpus
        self.classes = classes

    def _zero_prior(self):
        self._predictions, self._prior,_ = self.model.predict_batch(self.corpus.x, return_scores=True)

    def _sample(self, k):
        self.sample_data = self.strategy(dataset=self.corpus, prior=self._prior, embedding=self._embedding, k=k)
        self.dataset = self.create_dataset(self.sample_data)
        self.train, self.valid = validation_split(self.dataset, fraction=self._valid_split)

    def create_dataset(self, data, target, classes):
        print("This would be where the annotation fun happens")
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        train = self.sample_data
        print(train.count())
        return self.model.fit(train=train, valid=self.valid, *args, **kwargs)

    def reset(self):
        self.model = self._create_model()
        self.model.create_labels(self.classes)
        self.sample_data = None

    def _embed(self):
        _, self._embedding = self.model.embed(self.corpus)

    def pretrain(self, **kwargs):
        self._pretrain = True
        kwargs["log_mlflow"] = self._log_mlflow
        kwargs["valid_prefix"] = "pretrain"
        self._pretrain_kwargs = kwargs

    def active_transfer(self, *args, k,  **kwargs):
        self.reset()
        if self._pretrain:
            self.model.create_labels(self._pretrain_kwargs["train"].classes)
            self.model.fit(**self._pretrain_kwargs)
        if "prior" in self.strategy._sampler_kwargs: self._zero_prior()
        if "embedding" in self.strategy._sampler_kwargs: self._embed()

        self._sample(k)

        self.model.create_labels(self.classes)
        _, e = self.model.evaluate(self.valid)
        if self._log_mlflow: e.log_mlflow(prefix="zeroshot")
        print(e)
        return self.fit( *args, **kwargs)

    def evaluate_setting(self, runs=5, *args, **kwargs):
        histories = []
        for run in range(runs):
            if runs > 1: print(f"---------------------------\nRun {run}\n")
            histories.append(self.active_transfer(*args, **kwargs))
        return histories

    def evaluate_setting_k(self, n, *args, **kwargs):
        self.all_history = []
        for k in n:
            print("###################################################")
            print(f"Sample size {k}")
            self.all_history.append(self.evaluate_setting(k=k, *args, **kwargs))
            print("###################################################")



# import mlmc_lab.mlmc_experimental as mlmce
# import mlmc
# def get_model():
#     return mlmc.models.zeroshot.embedding.Siamese(
#         representation="sentence-transformers/paraphrase-mpnet-base-v2",
#         classes = d["classes"], target="single", finetune="all",
#         sformatter=mlmce.data.SFORMATTER["agnews"],
#         device="cuda:0")
#
# d = mlmce.data.get("agnews")
# t = Transfer(get_model, "random", log_mlflow=False)
# # t.pretrain(train = mlmc.data.sampler(mlmce.data.get("dbpedia")["train"], absolute=500))
# t.set_dataset(d["train"].x, d["classes"])
# t.evaluate_setting_k(n=[10,20,40,100], runs = 2, epochs=10)
