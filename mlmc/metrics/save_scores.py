import pathlib
import numpy as np
import datetime
import torch

class SaveScores:
    def __init__(self, output="."):
        self.output=pathlib.Path(output)
        if not self.output.exists(): self.output.mkdir(parents=True)
        self.counter = 0
        self.some_hash = str(hash(datetime.datetime.now()))[1:7]
        self.reset()

    def init(self, classes, target, **kwargs):
        "an extra function for model specific parameters of the metric"
        self.counter +=1
        self._zeroshot_ind = kwargs["_zeroshot_ind"]
        self.filename = self.output/f"{self.counter}-{self.some_hash}_scores.npy"
        self.reset()

    def reset(self):
        self.truth = []
        self.pred = []

    def update(self, batch):
        self.truth.append(batch[1])
        self.pred.append(batch[0])

    def compute(self):
        np.save(
            self.filename,
            {
                "truth": torch.cat(self.truth,0),
                "scores": torch.cat(self.pred,0),
                "_zeroshot_ind": self._zeroshot_ind
            }
        )
        return str(self.filename)

    def print(self):
        return str(self.filename)


