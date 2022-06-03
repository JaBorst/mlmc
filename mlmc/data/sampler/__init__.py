from .strategies import random_sampler, max_margin_sampler
from .class_based_strategies import class_medoid_sampler,class_uncertainty_sampler, class_certainty_sampler

register = {
    "max_margin": max_margin_sampler,
    "random": random_sampler,
}

class Sampler:
    def __init__(self, sampler):
        self.sampler = sampler
        self._sampler_kwargs = self.sampler.__code__.co_varnames[1:self.sampler.__code__.co_argcount]

    def __call__(self, dataset, **kwargs):
        curr_args = {x:kwargs[x] for x in self._sampler_kwargs if x != "self"}
        if isinstance(dataset, list):
            return self._corpus_sample(dataset, **curr_args)
        else:
            return self._dataset_sample(dataset, **curr_args)

    def _dataset_sample(self, data, **kwargs):
        idx = self.sampler(list(range(len(data))), **kwargs)
        return type(data)(x=[data.x[i] for i in idx], y=[data.y[i] for i in idx], classes=data.classes)

    def _corpus_sample(self, data: list, **kwargs):
        idx = self.sampler(list(range(len(data))), **kwargs)
        return [data[x] for x in idx]



def get(x):
    return Sampler(register[x])
