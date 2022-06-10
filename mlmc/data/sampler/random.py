"""
A collection of sampling methods for the MultilabelDataset object

.
"""
import numpy as np
import torch

def sampler(dataset, fraction=None, absolute=None):
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
    if fraction is not None:
        assert fraction<=1 and fraction>0, "The fraction argument has to be between 0 and 1."
    n_samples = absolute if absolute is not None else  int(fraction*len(dataset))
    if n_samples > len(dataset): n_samples = len(dataset)
    ind = np.random.choice(range(len(dataset)), n_samples, replace=False)
    x = [dataset.x[i] for i in ind]
    y = [dataset.y[i] for i in ind]
    return type(dataset)(x=x, y=y, classes=dataset.classes, target_dtype=dataset.target_dtype)

def subset(dataset, index):
    if isinstance(index[0], bool) or isinstance(index[0].item(), bool):
        index = np.where(index)[0]
    x = [dataset.x[i] for i in index]
    y = [dataset.y[i] for i in index]
    return type(dataset)(x=x, y=y, classes=dataset.classes, target_dtype=dataset.target_dtype)

def successive_sampler(dataset, classes, separate_dataset, reindex_classes=True):
    """
    Return an iterable of datasets sampled from dataset.

    Args:
         dataset: The input dataset (MultilabelDataset)
         classes: A classes mapping
         separate_dataset: The number of datasets to generate
         reindex_classes: If True, classes in the subsampled datasets are reindexed to 1:len(classes)
    Returns:
        A list of MultilabelDatasets
    """

    # Quick FIX function was changing the global value of mutable
    # ToDO: Find a better way maybe than copying
    classes=classes.copy()

    n_result = []
    n_idx = []
    n_ind = []
    already_select_id = set()
    already_select_class = {}
    n_samples =  np.round(len(dataset)/separate_dataset).astype(np.int64)

    for x in range(0, separate_dataset):
        # extend classes to use
        # sample from existing classes/delete from whole dataset to sample
        c_ind = np.random.choice(range(len(classes)), np.round(len(classes)/2).astype(np.int64))
        selectedKeys = list()
        # Iterate over the dict and put to be deleted keys in the list
        for index, (key, value) in enumerate(classes.items()):
            if(index in c_ind):
                print(index, key, value)
                already_select_class[key] = value
                selectedKeys.append(key)
 
        #Iterate over the list and delete corresponding key from dictionary
        for key in selectedKeys:
            if key in classes :
                del classes[key]
        l_list = dataset.y
        
        #Object to store which data point belongs to already defined classes
        candidate_idx = []

        for index, (c_list) in enumerate(dataset.y):
            intersect = set(c_list).intersection(set(already_select_class.keys()))
            if len(intersect) > 0:
                candidate_idx.append(index)
                l_list[index] = list(intersect)

        ind = np.random.choice(list(set(candidate_idx) - already_select_id), n_samples)

        already_select_id = set(ind).union(already_select_id)
        x = [dataset.x[i] for i in list(already_select_id)]
        y = [l_list[i] for i in list(already_select_id)]

        x_new = [dataset.x[i] for i in ind]
        y_new = [l_list[i] for i in ind]

        # Quick FIX function was changing the global value of mutable
        # ToDO: Find a better way maybe than copying
        cls = dict(zip(already_select_class.keys(), range(len(already_select_class)))) if reindex_classes else already_select_class.copy()

        n_result.append({
            'train' : type(dataset)(x=x, y=y, classes=dataset.classes, occuring_classes=cls, target_dtype=dataset.target_dtype),
            'test':type(dataset)(x=x_new, y=y_new, classes=dataset.classes, occuring_classes=cls,target_dtype=dataset.target_dtype)
        })
        n_idx.append(list(already_select_id))

        print("--- LENGTH DATASET: ",len(already_select_id))

    return n_result,n_idx


def class_sampler(data, classes, samples_size):
    """
    Create a subset of the data containing only classes in the classes argument and subsampling each class
    roughly to the number given in sample_size.

    Args:
        data: MultilabelDataset to sample from
        classes: subset of classes
        samples_size: Number of examples for each class (roughly)
    Returns:
        A dataset with roughly sample_size number of instances per class in classes.
    """
    if isinstance(samples_size, int):
        n = [samples_size for _ in range(len(classes))]
    else:
        n = samples_size

    index_list = list(range(len(data.x)))
    import random
    random.shuffle(index_list)

    new_data = []
    new_labels = []
    size_counter = [0 for _ in range(len(classes))]

    for i in index_list:
        example = data.x[i]
        labels = data.y[i]
        if any([x in classes for x in labels]):
            current_labels = [x for x in labels if x in classes]

            fraction_of_needed_samples = sum([size_counter[classes.index(x)] for x in current_labels]) /\
                                         (1+sum([min(n[classes.index(x)], size_counter[classes.index(x)] ) for x in current_labels]))

            if fraction_of_needed_samples <=1:
                new_data.append(example)
                new_labels.append(current_labels)

                for c in current_labels:
                    size_counter[classes.index(c)] = 1 + size_counter[classes.index(c)]

        if all([True if i>=j else False for i,j in zip(size_counter,n)]):
            break

    return type(data)(x=new_data, y=new_labels, classes=data.classes, occurring_classes={k: data.classes[k] for k in classes})



def validation_split(dataset, fraction=None, absolute=None):
    """
    Split a dataset into two separate splits (validation split)

    Args:
        dataset: data set to split
        fraction: The fraction of the data that should be returned (0<fraction<1)
        absolute: The absolute size of the sampled dataset
    Returns:
         A tuple of randomly subsampled MultilabelDatasets of the desired size.
    """
    # assert fraction is None != absolute is None, "Exactly one of fraction or absolute has to be set."
    if fraction is not None:
        assert fraction < 1 and fraction > 0, "The fraction argument has to be between 0 and 1."
    ind = list(range(len(dataset)))
    np.random.shuffle(ind)
    n_samples = (len(dataset)-absolute) if absolute is not None else int((1-fraction) * len(dataset))
    return type(dataset)(x=[dataset.x[i] for i in ind[:n_samples]],
                         y=[dataset.y[i] for i in ind[:n_samples]],
                         classes=dataset.classes), \
           type(dataset)(x=[dataset.x[i] for i in ind[n_samples:]],
                         y=[dataset.y[i] for i in ind[n_samples:]],
                         classes=dataset.classes)

def kfolds(dataset, k=10):
    from sklearn.model_selection import KFold
    for i1, i2 in KFold(k, shuffle=True).split(dataset.y):
        yield {"train":
            type(dataset)(
                x = [i for n, i in enumerate(dataset.x) if n in i1],
                y = [i for n, i in enumerate(dataset.y) if n in i1],
                classes=dataset.classes
            ),
        "test":
            type(dataset)(
                x=[i for n, i in enumerate(dataset.x) if n in i2],
                y=[i for n, i in enumerate(dataset.y) if n in i2],
                classes=dataset.classes
            )
        }


import random
def fewshot_sampler(dataset, k):
    from ..dataset_classes import is_multilabel
    if is_multilabel(dataset):
        index = list(range(len(dataset)))
        random.shuffle(index)
        sample_x = []
        sample_y = []
        counter = {k: 0 for k in dataset.classes}

        freq = dataset.count()
        dist = sorted(freq.items(), key=lambda x: x[1])

        for l, f in dist:
            if f == 0:
                continue
            # else:  break
            e = [(x,y) for (x,y) in zip(dataset.x,dataset.y) if l in y]
            w = [1/sum([freq[l] for l in y]) for x,y in e]
            e = random.choices(e,weights=w, k=min(len(e),k),)

            sample_x.extend([x[0] for x in e])
            sample_y.extend([x[1] for x in e])

            for lset in [x[1] for x in e]:
                for l in lset:
                    counter[l] = counter[l] + 1
            if all([k <= v for v in counter.values()]):
                break

        for key, val in counter.items():
            if val==0:
                for x, y in zip(dataset.x, dataset.y):
                    if key in y:
                        sample_x.append(x)
                        sample_y.append(y)
                        for al in y:
                            counter[al] = counter[al] + 1
                        if counter[key] >= k:
                            break
        data = type(dataset)(x=sample_x, y=sample_y, classes=dataset.classes)
        data.count(list(data.classes.keys()))
        return data
    else:
        index = list(range(len(dataset)))
        random.shuffle(index)
        sample_x = []
        sample_y = []
        counter = {k: 0 for k in dataset.classes}
        for i in index:
            if counter[dataset.y[i][0]] < k:
                sample_x.append(dataset.x[i])
                sample_y.append(dataset.y[i])
                counter[dataset.y[i][0]] = counter[dataset.y[i][0]] + 1

            if all([k <= v for v in counter.values()]):
                break
        return type(dataset)(x = sample_x, y = sample_y, classes = dataset.classes)

def entailment_split(dataset, fraction=None, absolute=None):
    """
    Split a dataset into two separate splits (validation split)

    Args:
        dataset: data set to split
        fraction: The fraction of the data that should be returned (0<fraction<1)
        absolute: The absolute size of the sampled dataset
    Returns:
         A tuple of randomly subsampled MultilabelDatasets of the desired size.
    """
    # assert fraction is None != absolute is None, "Exactly one of fraction or absolute has to be set."
    if fraction is not None:
        assert fraction < 1 and fraction > 0, "The fraction argument has to be between 0 and 1."
    ind = list(range(len(dataset)))
    np.random.shuffle(ind)
    n_samples = (len(dataset)-absolute) if absolute is not None else int((1-fraction) * len(dataset))
    return type(dataset)(x1=[dataset.x1[i] for i in ind[:n_samples]],
                         x2=[dataset.x2[i] for i in ind[:n_samples]],
                         labels=[dataset.labels[i] for i in ind[:n_samples]],
                         classes=dataset.classes), \
           type(dataset)(x1=[dataset.x1[i] for i in ind[n_samples:]],
                         x2=[dataset.x2[i] for i in ind[n_samples:]],
                         labels=[dataset.labels[i] for i in ind[n_samples:]],
                         classes=dataset.classes)

