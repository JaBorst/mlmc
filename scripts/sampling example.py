import mlmc

data = mlmc.data.get_multilabel_dataset("blurbgenrecollection")

data.keys()

def sample(dataset, fraction=None, absolute=None):


    from numpy.random import choice
    n_samples = absolute if absolute is not None else  int(fraction*len(dataset))
    ind = choice(range(len(dataset)), n_samples)
    x = [dataset.x[i] for i in ind]
    y = [dataset.y[i] for i in ind]


    return type(dataset)(x=x, y=y, classes=dataset.classes, target_dtype=dataset.target_dtype)

print(len(data["train"]))
print(len(sample(data["train"], absolute=1000)))