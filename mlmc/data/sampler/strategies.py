import numpy as np

def random_sampler(idx, k=None):
    """
    Sample a Random subsample of fixed size or fixed fraction of a dataset (i.i.d. sample).
    Args:
        dataset: A instance of mlmc.data.MultilabelDataset
        fraction: The fraction of the data that should be returned (0<fraction<1)
        absolute: The absolute size of the sampled dataset
    Returns:
         A randomly subsampled MultilabelDataset of the desired size.
    """

    # assert fraction is None != absolute is None, "Exactly one of fraction or absolute has to be set."
    assert k < len(idx)
    ind = np.random.choice(idx, k, replace=False)
    return ind


def max_margin_sampler(idx, k, prior):
    v, i = prior.sort(-1)
    margin = v[:, -1] - v[:, -2]
    ex = margin.sort()[1][-k:]
    return [idx[i] for i in ex]
