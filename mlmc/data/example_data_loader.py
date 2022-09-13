from ..data import _load_from_tmp,_save_to_tmp
from pathlib import Path
import tempfile
from urllib.request import urlopen
import tarfile


def load_agnews():
    """
    Loads AG News dataset from cache. If it doesn't exist in cache the dataset will be downloaded.
    Task: Singlelabel Classification

    :return: Tuple of form (data, classes)
    """
    url = "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz"
    data = _load_from_tmp("agnews")
    if data is not None:
        return data
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            resp = urlopen(url)
            tf = tarfile.open(fileobj=resp, mode="r|gz")
            tf.extractall(Path(tmpdir))
            testdir = Path(tmpdir) / 'ag_news_csv/test.csv'
            traindir = Path(tmpdir) / 'ag_news_csv/train.csv'
            classesdir = Path(tmpdir) / 'ag_news_csv/classes.txt'

            with open(testdir, "r") as f:
                testdata = [x.replace("\n", "").split('","')[::-1] for x in f.readlines()]
                testlabel = [int(x[2].replace('"', '')) for x in testdata]
                testtitle = [x[1] for x in testdata]
                testdescription = [x[0] for x in testdata]
                testtext = [" \n ".join([t, d]) for t, d in zip(testtitle, testdescription)]
            with open(traindir, "r") as f:
                traindata = [x.replace("\n", "").split('","')[::-1] for x in f.readlines()]
                trainlabel = [int(x[2].replace('"', '')) for x in traindata]
                traintitle = [x[1] for x in traindata]
                traindescription = [x[0] for x in traindata]
                traintext = [" \n ".join([t, d]) for t, d in zip(traintitle, traindescription)]
            with open(classesdir,"r") as f:
                classes = [x.replace("\n","") for x in f.readlines()]
                classes = dict(zip(classes, range(len(classes))))
                rev_classes = {v: k for k, v in classes.items()}
        data = {
            "train": (traintext,[[rev_classes[x-1]] for x in trainlabel]),
            "test": (testtext, [[rev_classes[x-1]] for x in testlabel]),
            "test_title": testtitle,
            "test_description": testdescription,
            "train_title": traintitle,
            "train_description": traindescription
        }
        _save_to_tmp("agnews", (data, classes))
        return data, classes
