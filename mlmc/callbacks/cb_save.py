from .abstract import Callback
import pathlib
import torch

class CallbackSaveAndRestore(Callback):
    def __init__(self, path, file="model", metric="valid_loss", mode="min"):
        super(CallbackSaveAndRestore, self).__init__()
        self.name = "callback_saveandrestore"
        self.dir = pathlib.Path(path)
        if not self.dir.exists(): self.dir.mkdir(parents=True)
        self.file = file
        self.metric = metric.split(".")
        self.mode = mode
        self.training_procedure = []

    def _add_metric(self, m):
        tmp = m.validation[-1]
        for n in self.metric:
            tmp = tmp[n]
        self.training_procedure.append(tmp)

    def on_epoch_end(self, model):
        torch.save(model.state_dict(), self.dir / f"{self.file}_{len(self.training_procedure)-1}.pt")
        self._add_metric(model)

    def on_train_end(self, model):
        i = torch.argmax(torch.tensor(self.training_procedure)) if self.mode=="max" else torch.argmin(torch.tensor(self.training_procedure))
        print(f"Restoring epoch {i+1}")
        model.load_state_dict(torch.load(self.dir / f"{self.file}_{i}.pt"))