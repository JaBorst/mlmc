import torch

from .abstract import Callback

class CallbackNLA(Callback):
    def __init__(self, threshold=0.9, target_threshold=0.5, difference=False, quiet=False, epochs=None):
        super(CallbackNLA, self).__init__()
        self.name = "NLA"
        self.threshold = threshold
        self.difference = difference
        self.target_threshold = target_threshold
        self.quiet = quiet
        self.epochs = epochs


    def on_epoch_end(self, model):
        labels, scores, _ = model.predict_dataset(model.train_data_set, return_scores=True, pbar=False)
        scores = scores.max(-1)[0].cpu()
        keep = [not(x!=y and scores[i].item()>self.threshold) for i,(x,y) in enumerate(zip(labels, model.train_data_set.y))]

        new_data = type(model.train_data_set)(x=[x for x,k in zip(model.train_data_set.x, keep) if k],
                                   y=[y for y, k in zip(model.train_data_set.y, keep) if k],
                                   classes = model.train_data_set.classes
                                   )


        if not self.quiet:
            from collections import Counter
            class_distr = Counter([y[0] for y, k in zip(model.train_data_set.y, keep) if not k])

            print(f"NLA removed {len(model.train_data_set)-len(new_data)} examples")
            print(class_distr)
        model.train_data_set = new_data

        if self.epochs is not None:
            self.threshold = self.threshold - (self.threshold-self.target_threshold)/self.epochs
        else:
            if self.threshold>self.target_threshold:
                self.threshold = self.threshold - (self.threshold-self.target_threshold)/10


    def on_train_end(self, model):
        pass
    def on_epoch_start(self, model):
        pass



class CallbackLNA(Callback):
    def __init__(self, threshold=0.9, target_threshold=0.5, difference=False, quiet=False, epochs=None):
        super(CallbackLNA, self).__init__()
        self.name = "LNA"
        self.threshold = threshold
        self.difference = difference
        self.target_threshold = target_threshold
        self.quiet = quiet
        self.epochs = epochs


    def on_epoch_end(self, model):
        labels, scores, _ = model.predict_dataset(model.train_data_set, return_scores=True, pbar=False)
        with torch.no_grad():
            l = model.loss.loss(scores, torch.tensor([x["labels"] for x in model.train_data_set]).to(scores.device)).cpu()
            mask = (l < (l.mean() + 1 * l.std())) #& (l > l.mean() - 1 * l.std())
            mask = mask.cpu().tolist()



        new_data = type(model.train_data_set)(x=[x for x,k in zip(model.train_data_set.x, mask) if k],
                                   y=[y for y, k in zip(model.train_data_set.y, mask) if k],
                                   classes = model.train_data_set.classes
                                   )


        if not self.quiet:
            from collections import Counter
            class_distr = Counter([y[0] for y, k in zip(model.train_data_set.y, mask) if not k])

            print(f"LNA removed {len(model.train_data_set)-len(new_data)} examples")
            print(class_distr)
        model.train_data_set = new_data


    def on_train_end(self, model):
        pass
    def on_epoch_start(self, model):
        pass
