from .abstract import Callback
import pathlib
import torch
import shutil
import tempfile

class CallbackSaveAndRestore(Callback):
    def __init__(self, path=None, file="model", metric="valid_loss", mode="min", delete_after=True):
        super(CallbackSaveAndRestore, self).__init__()
        self.name = "callback_saveandrestore"
        if path is None:
            self._using_tempdir=True
            self.tmpdir = tempfile.TemporaryDirectory()
            self.dir = pathlib.Path(str(self.tmpdir.name))
            if not delete_after: Warning("If your not specifying a path the temporary directory will always be removed after training. `delete_after` has no effect.")
        else:
            self._using_tempdir=False
            self.dir = pathlib.Path(path)
            if not self.dir.exists(): self.dir.mkdir(parents=True)
        self.file = file
        self.metric = metric.split(".")
        self.mode = mode
        self.training_procedure = []
        self.delete_after = delete_after

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

        if self._using_tempdir:
            self.tmpdir.cleanup()
        elif self.delete_after:
            shutil.rmtree(self.dir)