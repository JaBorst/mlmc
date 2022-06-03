from mlmc.data.sampler import get

class TransferTest:
    def __init__(self, model, strategy, log_mlflow=False, augmenter=None):
        assert callable(model), "model should be a callable creating the model"
        self._create_model = model
        self.strategy = get(strategy)
        self._prior=None
        self._embedding=None
        self._pretrain=False
        self._log_mlflow = log_mlflow
        self._augmenter = augmenter


    def set_dataset(self, train, valid=None, test=None):
        self.data_train = train
        self.data_valid = valid
        self.data_test = test
        self.classes = self.data_train.classes

    def _zero_prior(self):
        self._predictions, self._prior,_ = self.model.predict_batch(self.data_train.x, return_scores=True)

    def _sample(self, k):
        self.sample_data = self.strategy(dataset=self.data_train, prior=self._prior, embedding=self._embedding, k=k)


    def fit(self, *args, **kwargs):
        train = self.sample_data
        train.set_augmenter(self._augmenter)
        print(train.count())
        return self.model.fit(train=train, valid=self.data_valid, *args, **kwargs)

    def reset(self):
        self.model = self._create_model()
        self.model.create_labels(self.classes)
        self.sample_data = None

    def _embed(self):
        _, self._embedding = self.model.embed(self.data_train)

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

        self.model.create_labels(self.classes)
        _, e = self.model.evaluate(self.data_valid)
        if self._log_mlflow: e.log_mlflow(prefix="zeroshot")
        print(e)
        self._sample(k)
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

