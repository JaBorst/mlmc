from numpy.random import choice


def sampler(dataset, fraction=None, absolute=None):
    '''Sample a Random subsample of fixed size or fixed fraction of a dataset.'''
    n_samples = absolute if absolute is not None else  int(fraction*len(dataset))
    ind = choice(range(len(dataset)), n_samples)
    x = [dataset.x[i] for i in ind]
    y = [dataset.y[i] for i in ind]
    return type(dataset)(x=x, y=y, classes=dataset.classes, target_dtype=dataset.target_dtype)


def successive_sampler(dataset):
    """Return an iterable of datasets sampled from dataset... """
    #ToDo:
    # - Implementation:)
    return None