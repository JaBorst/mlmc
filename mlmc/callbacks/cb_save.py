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
            if not self.dir.exists(): self.dir.mkdir(parents=True, exist_ok=True)
        self.file = file
        self.metric = metric.split(".")
        self.mode = mode
        self.training_procedure = []
        self.delete_after = delete_after
        self._restore_index = -1

    def _add_metric(self, m):
        tmp = m.validation[-1]
        for n in self.metric:
            tmp = tmp[n]
        self.training_procedure.append(tmp)

    def on_epoch_end(self, model):
        self._add_metric(model)
        overwrite = False
        if self.mode == "max":
            if max(self.training_procedure) == self.training_procedure[-1]:
                overwrite = True
        else:
            if min(self.training_procedure) == self.training_procedure[-1]:
                overwrite = True
        if overwrite:
            self._restore_index = len(self.training_procedure)-1
            for f in self.dir.glob('*.pt'):
                try:
                    f.unlink()
                except OSError as e:
                    print("Error: %s : %s" % (f, e.strerror))
            torch.save(model.state_dict(), self.dir / f"{self.file}_{len(self.training_procedure)-1}.pt")

    def on_train_end(self, model):
        print(f"Restoring epoch {self._restore_index+1}")
        model.load_state_dict(torch.load(self.dir / f"{self.file}_{self._restore_index}.pt"))

        if self._using_tempdir:
            self.tmpdir.cleanup()
        elif self.delete_after:
            shutil.rmtree(self.dir)