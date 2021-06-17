from .data_loaders_classification import _load_from_tmp,_save_to_tmp
from urllib import error
from urllib.request import urlopen
from zipfile import ZipFile

from io import BytesIO
import tarfile
import tempfile

def load_sts12():
    data = _load_from_tmp(f"sts12")
    if data is not None:
        return data
    else:
        sts12_train = "https://www.cs.york.ac.uk/semeval-2012/task6/data/uploads/datasets/train.tgz"
        sts12_test = "https://www.cs.york.ac.uk/semeval-2012/task6/data/uploads/datasets/test-gold.tgz"

        with tempfile.TemporaryDirectory() as t:
            try:
                resp = urlopen(sts12_train)
            except error.HTTPError:
                print(error.HTTPError)
                return None
            assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)

            with open(t + "/tmp", 'wb') as f:
                f.write(resp.read())
            with tarfile.open(t + "/tmp","r:*") as f:
                d = sum([
                    [x.decode()[:-1].split("\t")  for x in f.extractfile("train/STS.input.SMTeuroparl.txt").readlines()],
                    [x.decode()[:-1].split("\t") for x in f.extractfile("train/STS.input.MSRvid.txt").readlines()],
                    [x.decode()[:-1].split("\t")  for x in f.extractfile("train/STS.input.MSRpar.txt").readlines()]
                    ],[])

                l = sum([
                    [float(x.decode()[:-1]) for x in f.extractfile("train/STS.gs.SMTeuroparl.txt").readlines()],
                    [float(x.decode()[:-1]) for x in f.extractfile("train/STS.gs.MSRvid.txt").readlines()],
                    [float(x.decode()[:-1]) for x in f.extractfile("train/STS.gs.MSRpar.txt").readlines()]]
                ,[])
            try:
                resp = urlopen(sts12_test)
            except error.HTTPError:
                print(error.HTTPError)
                return None
            assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)

            with open(t + "/tmp", 'wb') as f:
                f.write(resp.read())
            with tarfile.open(t + "/tmp","r:*") as f:
                d2 = sum([
                    [x.decode()[:-1].split("\t")  for x in f.extractfile("test-gold/STS.input.MSRpar.txt").readlines()],
                    [x.decode()[:-1].split("\t") for x in f.extractfile("test-gold/STS.input.MSRvid.txt").readlines()],
                    [x.decode()[:-1].split("\t")  for x in f.extractfile("test-gold/STS.input.SMTeuroparl.txt").readlines()],
                    [x.decode()[:-1].split("\t") for x in f.extractfile("test-gold/STS.input.surprise.OnWN.txt").readlines()],
                    [x.decode()[:-1].split("\t")  for x in f.extractfile("test-gold/STS.input.surprise.SMTnews.txt").readlines()]
                    ],[])

                l2 = sum([
                    [float(x.decode()[:-1]) for x in f.extractfile("test-gold/STS.gs.MSRpar.txt").readlines()],
                    [float(x.decode()[:-1]) for x in f.extractfile("test-gold/STS.gs.MSRvid.txt").readlines()],
                    [float(x.decode()[:-1]) for x in f.extractfile("test-gold/STS.gs.SMTeuroparl.txt").readlines()],
                    [float(x.decode()[:-1]) for x in f.extractfile("test-gold/STS.gs.surprise.OnWN.txt").readlines()],
                    [float(x.decode()[:-1]) for x in f.extractfile("test-gold/STS.gs.surprise.SMTnews.txt").readlines()]]
                ,[])


        data = {
            "train_x1": [x[0] for x in d],
            "train_x2": [x[1] for x in d],
            "train_y":  l,
             "test_x1": [x[0] for x in d2],
            "test_x2": [x[1] for x in d2],
            "test_y":  l2,
        }
        _save_to_tmp(f"sts12", data)
        return data



def load_sts14():
    data = _load_from_tmp(f"sts14")
    if data is not None:
        return data
    else:
        sts14_test ="http://alt.qcri.org/semeval2014/task10/data/uploads/sts-en-gs-2014.zip"

        with tempfile.TemporaryDirectory() as t:
            try:
                resp = urlopen(sts14_test)
            except error.HTTPError:
                print(error.HTTPError)
                # return None
            assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)

            with  ZipFile(BytesIO(resp.read())) as zf:

                d = sum([
                    [x.decode()[:-1].split("\t")  for x in zf.open("sts-en-test-gs-2014/STS.input.images.txt").readlines()],
                    [x.decode()[:-1].split("\t")  for x in zf.open("sts-en-test-gs-2014/STS.input.OnWN.txt").readlines()],
                    [x.decode()[:-1].split("\t")  for x in zf.open("sts-en-test-gs-2014/STS.input.tweet-news.txt").readlines()],
                    [x.decode()[:-1].split("\t")  for x in zf.open("sts-en-test-gs-2014/STS.input.deft-news.txt").readlines()],
                    [x.decode()[:-1].split("\t")  for x in zf.open("sts-en-test-gs-2014/STS.input.deft-forum.txt").readlines()],
                    [x.decode()[:-1].split("\t")  for x in zf.open("sts-en-test-gs-2014/STS.input.headlines.txt").readlines()]
                    ],[])

                l = sum([
                    [float(x.decode()[:-1]) for x in zf.open("sts-en-test-gs-2014/STS.gs.images.txt").readlines()],
                    [float(x.decode()[:-1]) for x in zf.open("sts-en-test-gs-2014/STS.gs.OnWN.txt").readlines()],
                    [float(x.decode()[:-1]) for x in zf.open("sts-en-test-gs-2014/STS.gs.tweet-news.txt").readlines()],
                    [float(x.decode()[:-1]) for x in zf.open("sts-en-test-gs-2014/STS.gs.deft-news.txt").readlines()],
                    [float(x.decode()[:-1]) for x in zf.open("sts-en-test-gs-2014/STS.gs.deft-forum.txt").readlines()],
                    [float(x.decode()[:-1]) for x in zf.open("sts-en-test-gs-2014/STS.gs.headlines.txt").readlines()]
                    ]
                ,[])

            # Also load the 2013 version
                with open(t + "/tmp", 'wb') as f:
                    f.write(zf.open("sts-en-test-gs-2014/sts2013-test.tgz").read())
            with tarfile.open(t + "/tmp", "r:*") as f:
                with tarfile.open(t + "/tmp", "r:*") as f:
                    d2 = sum([
                        [x.decode()[:-1].split("\t") for x in
                         f.extractfile("test-gs/STS.input.OnWN.txt").readlines()],
                        [x.decode()[:-1].split("\t") for x in
                         f.extractfile("test-gs/STS.input.headlines.txt").readlines()],
                        [x.decode()[:-1].split("\t") for x in
                         f.extractfile("test-gs/STS.input.FNWN.txt").readlines()],
                    ], [])

                    l2 = sum([
                        [float(x.decode()[:-1]) for x in f.extractfile("test-gs/STS.gs.OnWN.txt").readlines()],
                        [float(x.decode()[:-1]) for x in f.extractfile("test-gs/STS.gs.headlines.txt").readlines()],
                        [float(x.decode()[:-1]) for x in f.extractfile("test-gs/STS.gs.FNWN.txt").readlines()]
                        ]
                        , [])
            data13 = {
                "train_x1": [x[0] for x in d2],
                "train_x2": [x[1] for x in d2],
                "train_y": l2,
            }
            _save_to_tmp(f"sts13", data13)
        data = {
            "train_x1": [x[0] for x in d],
            "train_x2": [x[1] for x in d],
            "train_y":  l,
        }
        _save_to_tmp(f"sts14", data)
    return data


def load_sts13():
    data = _load_from_tmp(f"sts13")
    if data is not None:
        return data
    else:
        Warning("Please call load_sts14. This will autmatically generate sts13 aswell.")
    return None



def load_sts16():
    data = _load_from_tmp(f"sts16")
    if data is not None:
        return data
    else:

        sts16_test = "http://alt.qcri.org/semeval2016/task1/data/uploads/sts2016-english-with-gs-v1.0.zip"
        with tempfile.TemporaryDirectory() as t:
            try:
                resp = urlopen(sts16_test)
            except error.HTTPError:
                print(error.HTTPError)
                # return None
            assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)

            with  ZipFile(BytesIO(resp.read())) as zf:

                d = sum([
                    [x.decode()[:-1].split("\t") for x in
                     zf.open("sts2016-english-with-gs-v1.0/STS2016.input.answer-answer.txt").readlines()],
                    [x.decode()[:-1].split("\t") for x in
                     zf.open("sts2016-english-with-gs-v1.0/STS2016.input.headlines.txt").readlines()],
                    [x.decode()[:-1].split("\t") for x in
                     zf.open("sts2016-english-with-gs-v1.0/STS2016.input.plagiarism.txt").readlines()],
                    [x.decode()[:-1].split("\t") for x in
                     zf.open("sts2016-english-with-gs-v1.0/STS2016.input.postediting.txt").readlines()],
                    [x.decode()[:-1].split("\t") for x in
                     zf.open("sts2016-english-with-gs-v1.0/STS2016.input.question-question.txt").readlines()],
                ], [])

                l = sum([
                    [x.decode()[:-1] for x in zf.open("sts2016-english-with-gs-v1.0/STS2016.gs.answer-answer.txt").readlines()],
                    [x.decode()[:-1] for x in zf.open("sts2016-english-with-gs-v1.0/STS2016.gs.headlines.txt").readlines()],
                    [x.decode()[:-1] for x in zf.open("sts2016-english-with-gs-v1.0/STS2016.gs.plagiarism.txt").readlines()],
                    [x.decode()[:-1] for x in zf.open("sts2016-english-with-gs-v1.0/STS2016.gs.postediting.txt").readlines()],
                    [x.decode()[:-1] for x in zf.open("sts2016-english-with-gs-v1.0/STS2016.gs.question-question.txt").readlines()],
                ]
                    , [])

        index = [i for i, x in enumerate(l) if x != ""]
        d = [d[i] for i in index]
        l = [float(l[i]) for i in index]

        data = {
            "train_x1": [x[0] for x in d],
            "train_x2": [x[1] for x in d],
            "train_y":  l,
        }
        _save_to_tmp(f"sts16", data)
        return data

def load_stsb(lang="en"):
    from datasets import load_dataset
    train = load_dataset("stsb_multi_mt", name=lang, split="train")
    test = load_dataset("stsb_multi_mt", name=lang, split="test")
    dev = load_dataset("stsb_multi_mt", name=lang, split="dev")

    data = {
                "train_x1": train["sentence1"],
                "valid_x1":  dev["sentence1"],
                "test_x1":  test["sentence1"],
                "train_x2":  train["sentence2"],
                "valid_x2":  dev["sentence2"],
                "test_x2":  test["sentence2"],
                "train_y": train["similarity_score"],
                "valid_y": dev["similarity_score"],
                "test_y": test["similarity_score"]
            }
    return data


def load_sick():
    from datasets import load_dataset
    train = load_dataset("sick", split="train")
    test = load_dataset("sick", split="test")
    dev = load_dataset("sick", split="validation")

    data = {
                "train_x1": train["sentence_A"],
                "valid_x1":  dev["sentence_A"],
                "test_x1":  test["sentence_A"],
                "train_x2":  train["sentence_B"],
                "valid_x2":  dev["sentence_B"],
                "test_x2":  test["sentence_B"],
                "train_y": train["relatedness_score"],
                "valid_y": dev["relatedness_score"],
                "test_y": test["relatedness_score"]
            }
    return data

def load_sts():
    train_dataset = load_stsb()
    from .datasets import RegressionDataset
    train = RegressionDataset(x1=train_dataset["train_x1"], x2=train_dataset["train_x2"],
                                         labels=train_dataset["train_y"], )
    valid = RegressionDataset(x1=train_dataset["valid_x1"], x2=train_dataset["valid_x2"],
                                         labels=train_dataset["valid_y"], )
    test = RegressionDataset(x1=train_dataset["test_x1"], x2=train_dataset["test_x2"],
                                        labels=train_dataset["test_y"], )

    train_dataset = load_sts12()
    train12 = RegressionDataset(x1=train_dataset["train_x1"], x2=train_dataset["train_x2"],
                                           labels=train_dataset["train_y"], )
    test12 = RegressionDataset(x1=train_dataset["test_x1"], x2=train_dataset["test_x2"],
                                          labels=train_dataset["test_y"], )

    train_dataset = load_sts14()
    train14 = RegressionDataset(x1=train_dataset["train_x1"], x2=train_dataset["train_x2"],
                                           labels=train_dataset["train_y"], )

    train_dataset = load_sts13()
    train13 = RegressionDataset(x1=train_dataset["train_x1"], x2=train_dataset["train_x2"],
                                           labels=train_dataset["train_y"], )

    train_dataset = load_sts16()
    train16 = RegressionDataset(x1=train_dataset["train_x1"], x2=train_dataset["train_x2"],
                                           labels=train_dataset["train_y"], )

    complete  = train + valid +test + train12 + test12 + train14 + train13 + train16
    return complete
