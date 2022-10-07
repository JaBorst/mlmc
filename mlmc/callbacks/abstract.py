class Callback:
    def __init__(self):
        self.name = "Callback"

    def on_epoch_end(self, model):
        pass
    def on_train_end(self, model):
        pass
    def on_epoch_start(self, model):
        pass
    def on_batch_start(self, model):
        pass
    def on_batch_end(self, model):
        pass
