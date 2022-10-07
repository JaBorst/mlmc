from pathlib import Path
import pickle

CACHE = Path.home() / ".mlmc" / "datasets"

def _load_from_tmp(dataset):
    """
    Loads a dataset from cache.

    :param dataset: Name of the dataset
    :return: Tuple of form (data, classes) if dataset exists in cache, else None
    """
    if not Path.exists(CACHE):
        Path.mkdir(CACHE)
    if Path.is_file(CACHE / dataset):
        # print("Loading from cache...")
        with open(CACHE / dataset, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        return None

def _save_to_tmp(dataset, data):
    """
    Saves a dataset to cache.

    :param dataset: Name of the dataset
    :param data: Tuple of form (data, classes)
    :return: Path to the saved dataset if dataset didn't exist in cache, else False.
    """
    if not Path.exists(CACHE):
        Path.mkdir(CACHE)
    if not Path.is_file(CACHE / dataset):
        with open(CACHE / dataset, "wb") as f:
            pickle.dump(data, f)
        return CACHE / dataset
    else:
        return False

