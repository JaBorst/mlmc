"""
Defines dataset class and provides automated load/cache functions for some public multilabel datasets.
"""

import torch

from .data_loaders_classification import load_eurlex, load_wiki30k, load_huffpost, load_aapd, load_rcv1, \
    load_moviesummaries, load_blurbgenrecollection, load_blurbgenrecollection_de, load_20newsgroup, export, \
    load_agnews, load_dbpedia, load_ohsumed, load_yahoo_answers, load_movie_reviews, load_amazonfull, load_trec6, \
    load_trec50, load_yelpfull

from .data_loaders_similarity import load_sts, load_stsb, load_sts12, load_sts13, load_sts14, load_sts16, load_sick

# String Mappings
from .datasets import MultiLabelDataset, SingleLabelDataset, MultiOutputSingleLabelDataset, \
    MultiOutputMultiLabelDataset, RegressionDataset, PredictionDataset

register = {
    "aapd": load_aapd,
    "rcv1": load_rcv1,
    "huffpost": load_huffpost,
    "wiki30k": load_wiki30k,
    "eurlex": load_eurlex,
    "movies_summaries": load_moviesummaries,
    "blurbgenrecollection": load_blurbgenrecollection,
    "blurbgenrecollection_de": load_blurbgenrecollection_de,
    "20newsgroup": load_20newsgroup,
    "agnews": load_agnews,
    "dbpedia": load_dbpedia,
    "ohsumed": load_ohsumed,
    "yahoo_answers": load_yahoo_answers,
    "movie_reviews": load_movie_reviews,
    "trec6": load_trec6,
    "trec50":load_trec50,
    "yelpfull": load_yelpfull,
    "amazonfull": load_amazonfull
}


# -------------------------------------------------------------------------------------


def get_dataset(name, type, ensure_valid=False, valid_split=0.25, target_dtype=torch.FloatTensor):
    """
    General dataset getter for datasets in provided by the package.

    :param name: name of the dataset in register
    :param type: MultilabelDataset or SequenceDataset defined in mlmc.data
    :param ensure_valid: if True and there's no validation data in the original data a portion of the trainset is split
    :param valid_split: the fraction of the train set to be used as validation if ensure_valid=True
    :param target_dtype: Target Tensortype of the label multihot representation. (default torch.FloatTensor)
    :return: a dictionary with keys: "train", "valid" and "test" and additional information the dataset provides (graphs, maps, classes,..)
    """
    f = register.get(name, None)
    assert f is not None, "Dataset name not found"

    data, classes = f()

    if "valid" not in data.keys():
        # print("No Validation data found.")
        if ensure_valid:
            print("Providing random split...")
            from sklearn.model_selection import train_test_split
            splits_from_train = train_test_split(*data["train"], test_size=valid_split)
            train = [splits_from_train[0], splits_from_train[2]]
            valid = [splits_from_train[1], splits_from_train[3]]
            data["train"] = train
            data["valid"] = valid
        else:
            data["valid"] = None
    datasets = {
        split: type(x=data[split][0],
                    y=data[split][1],
                    classes=classes,
                    purpose=split,
                    target_dtype=target_dtype) if data[split] is not None else None
        for split in ["test", "train", "valid"]
    }
    for k in data.keys():
        if k not in ["test", "train", "valid"]:
            datasets[k] = data[k]
    datasets["classes"] = classes

    return datasets


## Wrapper for multilabel datasets
def get_multilabel_dataset(name):
    """
    Load multilabel training data if available.

    This is the default wrapper function for retrieving multilabel datasets.

    :param name: See: mlmc.data.register.keys()
    :return:
    """
    return get_dataset(name, type=MultiLabelDataset, ensure_valid=False, target_dtype=torch._cast_Float)


## Wrapper for singlelabel datasets
def get_singlelabel_dataset(name):
    """
    Load singlelabel training data if available.

    This is the default wrapper function for retrieving SingleLabelDataset datasets.
    This is a special case of MultilabelDataset.

    :param name: See: mlmc.data.register.keys()
    :return:
    """
    return get_dataset(name, type=SingleLabelDataset, ensure_valid=False, target_dtype=torch._cast_Float)

def get(name):
    """
    Universal get function for datasets.

    :param name: Name of the dataset
    :return: A dataset if the name exists
    """
    try:
        try:
            return get_singlelabel_dataset(name)
        except:
            return get_multilabel_dataset(name)
    except:
        print("Datset not found. Must be one of:")
        print(register.keys())

def is_multilabel(x):
    """
    Checks if input is a multilabel dataset.

    :param x: A dataset
    :return: True if multilabel, else False.
    """
    return type(x) in  (MultiLabelDataset, MultiOutputMultiLabelDataset)




## Sampler import
from .sampling_functions import sampler, successive_sampler, class_sampler, validation_split, kfolds, fewshot_sampler
from .dataset_formatter import  *