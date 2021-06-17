from mlmc.data.data_loaders_classification import _load_from_tmp,_save_to_tmp
import json
from urllib import error
from urllib.request import urlopen
from zipfile import ZipFile

from io import BytesIO


def load_mnli(binary=True):

    url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
    data = _load_from_tmp(f"mnli-{binary}")
    if data is not None:
        return data
    else:
        try:
            resp = urlopen(url)
        except error.HTTPError:
            print(error.HTTPError)
            return None
        assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)

        resp = urlopen(url)
        zf = ZipFile(BytesIO(resp.read()))
        with zf.open("multinli_1.0/multinli_1.0_train.jsonl") as f:
            lines = f.readlines()


            train_1 = []
            train_2 = []
            train_labels = []

            for line in lines:
                d = json.loads(line)
                train_1.append(d["sentence1"])
                train_2.append(d["sentence2"])
                train_labels.append(d["gold_label"])

            train_labels = [x if x != "-" else "contradiction" for x in train_labels]

        with zf.open("multinli_1.0/multinli_1.0_dev_matched.jsonl") as f:
            lines = f.readlines()

            valid_1 = []
            valid_2 = []
            valid_labels = []
            for line in lines:
                d = json.loads(line)
                valid_1.append(d["sentence1"])
                valid_2.append(d["sentence2"])
                valid_labels.append(d["gold_label"])
            valid_labels = [x if x != "-" else "contradiction" for x in valid_labels]
        classes ={"entailment":1, "neutral": 0, "contradiction":-1}
        if binary:
            classes = {"entailment": 1, "contradiction": -1}
            train_labels = [x if x == "entailment" else "contradiction" for x in train_labels]
            valid_labels = [x if x == "entailment" else "contradiction" for x in valid_labels]

        data = {
                    "train_x1": train_1,
                    "test_x1": valid_1,
                    "train_x2": train_2,
                    "test_x2": valid_2,
                    "train_y":train_labels,
                    "test_y":valid_labels
                }
        _save_to_tmp(f"mnli-{binary}", (data, classes))
        return data, classes



def load_snli(binary=True):

    url = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    data = _load_from_tmp(f"snli-{binary}")
    if data is not None:
        return data
    else:
        try:
            resp = urlopen(url)
        except error.HTTPError:
            print(error.HTTPError)
            return None
        assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)

        resp = urlopen(url)
        zf = ZipFile(BytesIO(resp.read()))
        with zf.open("snli_1.0/snli_1.0_train.jsonl") as f:
            lines = f.readlines()


            train_1 = []
            train_2 = []
            train_labels = []

            for line in lines:
                d = json.loads(line)
                train_1.append(d["sentence1"])
                train_2.append(d["sentence2"])
                train_labels.append(d["gold_label"])

            train_labels = [x if x != "-" else "contradiction" for x in train_labels]

        with zf.open("snli_1.0/snli_1.0_dev.jsonl") as f:
            lines = f.readlines()

            valid_1 = []
            valid_2 = []
            valid_labels = []
            for line in lines:
                d = json.loads(line)
                valid_1.append(d["sentence1"])
                valid_2.append(d["sentence2"])
                valid_labels.append(d["gold_label"])
            valid_labels = [x if x != "-" else "contradiction" for x in valid_labels]

        with zf.open("snli_1.0/snli_1.0_test.jsonl") as f:
            lines = f.readlines()

            test_1 = []
            test_2 = []
            test_labels = []
            for line in lines:
                d = json.loads(line)
                test_1.append(d["sentence1"])
                test_2.append(d["sentence2"])
                test_labels.append(d["gold_label"])
            test_labels = [x if x != "-" else "contradiction" for x in test_labels]

        classes ={"entailment":1, "neutral": 0, "contradiction":-1}
        if binary:
            classes = {"entailment": 1, "contradiction": -1}
            train_labels = [x if x == "entailment" else "contradiction" for x in train_labels]
            valid_labels = [x if x == "entailment" else "contradiction" for x in valid_labels]
            test_labels = [x if x == "entailment" else "contradiction" for x in test_labels]

        data = {
                    "train_x1": train_1,
                    "valid_x1": valid_1,
                    "test_x1": test_1,
                    "train_x2": train_2,
                    "valid_x2": valid_2,
                    "test_x2": test_2,
                    "train_y":train_labels,
                    "valid_y":valid_labels,
                    "test_y":test_labels
                }
        _save_to_tmp(f"snli-{binary}", (data, classes))
        return data, classes

