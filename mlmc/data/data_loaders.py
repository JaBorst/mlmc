"""A Collection of function to load, export and cache various datasets.
Repository for later use: http://manikvarma.org/downloads/XC/XMLRepository.html
"""

import json
import pickle
import zipfile
from pathlib import Path
from urllib import error
from urllib.request import urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

import networkx as nx
import tarfile
import tempfile
from bs4 import BeautifulSoup
from io import BytesIO
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CACHE = Path.home() / ".mlmc" / "datasets"
URL = "https://aspra29.informatik.uni-leipzig.de:9090/"


def _load_from_tmp(dataset):
    if not Path.exists(CACHE):
        Path.mkdir(CACHE)
    if Path.is_file(CACHE / dataset):
        print("Loading from cache...")
        with open(CACHE / dataset, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        return None


def _save_to_tmp(dataset, data):
    if not Path.exists(CACHE):
        Path.mkdir(CACHE)
    if not Path.is_file(CACHE / dataset):
        with open(CACHE / dataset, "wb") as f:
            pickle.dump(data, f)
        return True
    else:
        return False


# ============================================================================
# ============================================================================


def load_aapd():
    """Load AAPD Datasets"""
    data = _load_from_tmp("aapd")
    if data is not None:
        return data
    else:
        try:
            resp = urlopen(URL + "AAPD.zip")
        except error.HTTPError:
            print(error.HTTPError)
            return None
        assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)
        zf = ZipFile(BytesIO(resp.read()))
        with zf.open("AAPD/aapd.json") as f:
            classes = json.load(f)
        data = {}
        for x in ("test", "val", "train"):
            with zf.open("AAPD/text_" + x) as f:
                test_x = [x.decode() for x in f.readlines()]
            with zf.open("AAPD/label_" + x) as f:
                test_y = [[y.lower() for y in x.decode().replace("\n", "").split(" ")] for x in f.readlines()]
            if x == "val":
                data["valid"] = [test_x, test_y]
            else:
                data[x] = [test_x, test_y]
        _save_to_tmp("aapd", (data, classes))
    return data, classes


def load_rcv1(path=None):

    data = _load_from_tmp("rcv1")
    if data is not None:
        return data

    assert path is not None, "RCV1 must be input with a path to the downloaded corpus, when first called"
    dataset = Path(path)
    with open(dataset / "categories.txt")as f:
        labels = {}
        needed_zips = set()
        for x in f.readlines():
            if x.split()[0][-4:] == ".zip" and len(x.replace("\n", "").split()[2:]) > 0:
                needed_zips.add(x.split()[0])
                labels[x.split()[1]] = x.replace("\n", "").split()[2:]

    with open(dataset / "train.split") as f:
        train_ids = [x.replace("\n", "") for x in f.readlines()]
    with open(dataset / "test.split") as f:
        test_ids = [x.replace("\n", "") for x in f.readlines()]

    with open(dataset / "topic_codes.txt") as f:
        classes = [(x.split("\t")[0], x.replace("\n", "").split("\t")[1:]) for x in f.readlines()[2:-1]]
        classes = set([x[0] for x in classes])

    with tempfile.TemporaryDirectory() as tempdir:
        for file in tqdm(needed_zips):
            zipfile.ZipFile(dataset / file).extractall(Path(tempdir))

        def _get(ids):
            errors = 0
            documents = []
            labelsets = []
            for nid in tqdm(ids):
                file = nid + "newsML.xml"
                try:
                    with open(Path(tempdir) / file) as tmp:
                        xml = ElementTree.fromstring(tmp.read())
                        text = ElementTree.tostring(xml, method='text').decode().replace("\n", " ").replace("  ",
                                                                                                            " ").strip()
                        documents.append(text)
                        labelsets.append([x for x in labels[file] if x in classes])
                except Exception:
                    try:
                        with open(Path(tempdir) / file, encoding="iso-8859-1") as tmp:
                            xml = ElementTree.fromstring(tmp.read())
                            text = ElementTree.tostring(xml, method='text').decode().replace("\n", " ").replace("  ",
                                                                                                                " ").strip()
                            documents.append(text)
                            labelsets.append([x for x in labels[file] if x in classes])
                    except Exception as e2:
                        errors += 1
                        print(e2)
            print(str(errors) + " Errors")
            return (documents, labelsets)

        train = _get(train_ids)
        test = _get(test_ids)

    classes = list(classes.intersection(set([x for y in train[1] + test[1] for x in y])))
    classes = dict(zip(classes, range(len(classes))))
    data = {"train": train, "test": test}

    with open(dataset / "topic_codes.txt", "r") as f:
        topics = [x.replace("\n", "").split("\t") for x in f.readlines() if len(x) > 1][2:]
    topicmap = {x[0]: x[1] for x in topics}
    data["topicmap"] = topicmap

    with open(dataset / "rcv1.topics.hier.orig", "r") as f:
        content = f.readlines()
    import re
    edges = [(re.split(" +", x)[1], re.split(" +", x)[3]) for x in content]
    edges = [(topicmap.get(x[0],x[0]).capitalize(),topicmap.get(x[1],x[1]).capitalize()) for x in edges]
    graph = nx.DiGraph(edges[1:])
    data["graph"] = graph


    for key in ("train", "test"):
        data[key] = (data[key][0],[[topicmap[l].capitalize() for l in labellist] for labellist in data[key][1]])
    classes = {topicmap[k].capitalize(): v for k, v in classes.items()}

    data["topicmap"] = {v.capitalize():k for k,v in topicmap.items()}

    _save_to_tmp("rcv1", (data, classes))
    return data, classes


def load_wiki30k(path="/disk1/users/jborst/Data/Test/MultiLabel/wiki30k"):
    import pickle
    with open(Path(path) / "wiki30k_raw_text.p", "rb") as f:
        content = pickle.load(f)
    train_x = [x["text"] for x in content[0]]
    train_y = [x["catgy"] for x in content[0]]
    test_x = [x["text"] for x in content[1]]
    test_y = [x["catgy"] for x in content[1]]
    data = {"train": (train_x, train_y), "test": (test_x, test_y)}
    return data, content[3]


def load_eurlex():
    data = _load_from_tmp("eurlex")
    if data is not None:
        return data
    else:
        try:
            resp = urlopen(URL + "EURLex.zip")
        except error.HTTPError:
            print(error.HTTPError)
            return None
        assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)
        zf = ZipFile(BytesIO(resp.read()))
        with zf.open("EURLex/eurlex_raw_text.p") as f:
            content = pickle.load(f)
        train_x = [x["text"] for x in content[0]]
        train_y = [x["catgy"] for x in content[0]]
        test_x = [x["text"] for x in content[1]]
        test_y = [x["catgy"] for x in content[1]]
        classes = content[3]

        reverse_classes = {v: k for k, v in classes.items()}
        train_y = [[reverse_classes[x] for x in labels] for labels in train_y]
        test_y = [[reverse_classes[x] for x in labels] for labels in test_y]
        data = {}
        data["train"] = (train_x, train_y)
        data["test"] = (test_x, test_y)

        _save_to_tmp("eurlex", (data, classes))
    return data, classes


def load_huffpost(path=None, test_split=0.25):
    data = _load_from_tmp("huffpost")
    if data is not None:
        return data
    else:
        try:
            resp = urlopen(URL + "HuffPost.zip")
        except error.HTTPError:
            print(error.HTTPError)
            return None
        assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)
        zipfile = ZipFile(BytesIO(resp.read()))

        with zipfile.open("HuffPost/News_Category_Dataset_v2.json", "r") as f:
            content = json.loads("[" + ",".join([x.decode() for x in f.readlines()]) + "]")
        headlines = [x["headline"] for x in content]
        label = [x["category"] for x in content]
        data = (headlines, label)
        classes = list(set(label))
        classes = dict(zip(classes, range(len(classes))))
        n_arr = len(data)
        data = train_test_split(*data, test_size=test_split)
        tmp = [[], []]
        for i, arr in enumerate(data):
            tmp[i % n_arr].append(arr)

        _save_to_tmp("huffpost", (data, classes))

    return {"train": tmp[0], "test": tmp[1]}, classes


def load_moviesummaries(test_split=0.25):
    data = _load_from_tmp("moviesummaries")
    if data is not None:
        return data
    else:

        try:
            resp = urlopen(URL + "MovieSummaries.zip")
        except error.HTTPError:
            print(error.HTTPError)
            return None
        assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)
        zf = ZipFile(BytesIO(resp.read()))

        from sklearn.model_selection import train_test_split
        with zf.open("MovieSummaries/plot_summaries.txt", "r") as f:
            content = [x.decode() for x in f.readlines()]
        with zf.open("MovieSummaries/movie.metadata.tsv", "r") as f:
            meta = {x.decode().split("\t")[0]: x.decode().split("\t")[-1].replace("\n", "") for x in
                    f.readlines()}  # f.readlines()#[x.split("\t")[-1].replace("\n","") for x in f.readlines()]
        meta = {k: [x.split(": ")[-1].replace("\"", "").replace("}", "") for x in genre.split(", ")] for k, genre in
                meta.items()}
        data = [(x.split("\t")[1], meta[str(x.split("\t")[0])]) for x in content if
                str(x.split("\t")[0]) in meta.keys()]
        text = [x[0] for x in data]
        label = [x[1] for x in data]

        data = (text, label)
        classes = list(set([x for y in label for x in y]))
        classes = dict(zip(classes, range(len(classes))))

        n_arr = len(data)
        data = train_test_split(*data, test_size=test_split)
        tmp = [[], []]
        for i, arr in enumerate(data):
            tmp[i % n_arr].append(arr)

        dataset=  {"train": tmp[0], "test": tmp[1]}
        _save_to_tmp("moviesummaries", (dataset, classes))

        return dataset, classes


# ----------------------------------------------
# Hierarchical Multilabel
# see http://kt.ijs.si/DragiKocev/PhD/resources/doku.php?id=hmc_classification
# ----------------------------------------------

def load_blurbgenrecollection():
    url = "https://fiona.uni-hamburg.de/ca89b3cf/blurbgenrecollectionen.zip"
    data = _load_from_tmp("blurbgenrecollection")
    if data is not None:
        return data
    else:
        resp = urlopen(url)
        zf = ZipFile(BytesIO(resp.read()))
        data = {}
        for purpose in ["dev", "train", "test"]:
            soup = BeautifulSoup(
                "<root>" + zf.open("BlurbGenreCollection_EN_" + purpose + ".txt").read().decode("utf-8").replace(
                    "\n", "") + "</root>")
            text, labels = [], []
            for i in soup.findAll("book"):
                text.append(i.find("body").text)
                labels.append([x.text for x in i.find("topics").children])
            if purpose == "dev":
                data["valid"] = (text, labels)
            else:
                data[purpose] = (text, labels)

        edges = [x.decode("utf-8").replace("\n", "").split("\t") for x in zf.open("hierarchy.txt").readlines()]
        classes = list(set([x for y in data["train"][1] + data["valid"][1] + data["test"][1] + edges for x in y]))
        classes = dict(zip(classes, range(len(classes))))
        edges = [e for e in edges if len(e) == 2]
        graph = nx.DiGraph(edges)
        data["graph"] = graph
        _save_to_tmp("blurbgenrecollection", (data, classes))
        return data, classes


def load_blurbgenrecollection_de():
    url = "https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germeval-2019-hmc/germeval2019t1-public-data-final.zip"
    data = _load_from_tmp("blurbgenrecollection_de")
    if data is not None:
        return data
    else:

        resp = urlopen(url)
        zipfile = ZipFile(BytesIO(resp.read()))

        data = {}
        for purpose in ["dev", "train", "test"]:
            soup = BeautifulSoup(
                "<root>" + zipfile.open("blurbs_" + purpose + ".txt").read().decode(
                    "utf-8").replace("\n", "") + "</root>")
            text, labels = [], []
            for i in soup.findAll("book"):
                text.append(i.find("body").text)
                labels.append([x.text for x in i.find("categories").findAll("topic")])
            if purpose == "dev":
                data["valid"] = (text, labels)
            else:
                data[purpose] = (text, labels)

        edges = [x.decode("utf-8").replace("\n", "").split("\t") for x in zipfile.open("hierarchy.txt").readlines()]
        classes = list(set([x for y in data["train"][1] + data["valid"][1] + data["test"][1] + edges for x in y]))
        classes = dict(zip(classes, range(len(classes))))
        edges = [e for e in edges if len(e) == 2]
        graph = nx.DiGraph(edges)
        data["graph"] = graph
        _save_to_tmp("blurbgenrecollection_de", (data, classes))
        return data, classes


def load_webofscience():
    raise NotImplementedError
    # url = "https://data.mendeley.com/datasets/9rw3vkcfy4/6/files/c9ea673d-5542-44c0-ab7b-f1311f7d61df/WebOfScience.zip?dl=1"
    # data = _load_from_tmp("blurbgenrecollection")
    # if data is not None:
    #     return data
    # else:
    #     from xml.etree import ElementTree
    #     from urllib.request import urlopen
    #     from zipfile import ZipFile
    #     from io import BytesIO
    #     import re
    #
    #     resp = urlopen(url)
    #     zipfile = ZipFile(BytesIO(resp.read()))


################################
# Named Entity Recognition
################################
import re


def read_conll(file, column=3):
    """Read in the standard tab separated conll format"""
    with open(file, "r", encoding="utf-8") as file_handler:
        content = file_handler.read().strip("\n").strip()
        sentences = [[x.split(" ")[0].strip(" ").strip("\t") for x in sentence.split("\n")] for sentence in
                     re.split(r"\n\s*\n*\s*\n", content)]
        ner = [[x.split(" ")[column].strip(" ").strip("\t") for x in sentence.split("\n")] for sentence in
               re.split(r"\n\s*\n*\s*\n", content)]
    return sentences, ner


def load_conll2003en(path=None):
    """Load Conll2003 English Named Entity Dataset"""
    data = _load_from_tmp("conll2003en")
    if data is not None:
        return data
    else:
        assert path is not None, "Conll2003 path is None. Insert Path to conll2003 directory."
        path = Path(path)
        data = {}
        data["test"] = read_conll(path / "test.txt")
        data["valid"] = read_conll(path / "valid.txt")
        data["train"] = read_conll(path / "train.txt")
        from seq2.data.tagsets import NER
        classes = dict(zip(NER, range(len(NER))))
        _save_to_tmp("conll2003en", (data, classes))
    return data, classes


def load_20newsgroup():
    url = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
    data = _load_from_tmp("20newsgroup")
    if data is not None:
        return data
    else:

        with tempfile.TemporaryDirectory() as tmpdir:

            resp = urlopen(url)
            tf = tarfile.open(fileobj=resp, mode="r|gz")
            tf.extractall(Path(tmpdir))
            testdir = Path(tmpdir) / '20news-bydate-test'
            traindir = Path(tmpdir) / '20news-bydate-train'

            classes = []

            text, label = [], []
            for catg in testdir.iterdir():
                classes.append(catg.name)
                for file in catg.iterdir():
                    with open(file, 'r', encoding="ISO-8859-1") as f:
                        text.append(f.read())
                        label.append([catg.name])
            testdata = (text, label)

            text, label = [], []
            for catg in traindir.iterdir():
                classes.append(catg.name)
                for file in catg.iterdir():
                    with open(file, 'r', encoding="ISO-8859-1") as f:
                        text.append(f.read())
                        label.append([catg.name])
            traindata = (text, label)

            classes = list(set(classes))
            classes.sort()
            classes = dict(zip(classes, range(len(classes))))

            edges = [
                (x.split(".")[y], x.split(".")[y + 1] if y + 2 != len(x.split(".")) else x)
                for x in classes for y in range(len(x.split(".")) - 1)]
            edges += list(set([("ROOT", x.split(".")[0]) for x in classes]))

            graph = nx.DiGraph(edges)
            data = {"train": traindata, "test": testdata, "graph": graph}
        _save_to_tmp("20newsgroup", (data, classes))

        return data, classes

def load_agnews():
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

def export(data, classes, path=Path("./export")):
    path = Path(path)
    if not path.exists():
        path.mkdir()
    for k, v in data.items():
        if k in ("test", "valid", "train"):
            with open(path / k + "_x.txt", "w") as o:
                o.writelines([x.replace("\n", "\t") + "\n" for x in v[0]])
            with open(path / k + "_y.txt", "w") as o:
                o.writelines(["\t".join(x) + "\n" for x in v[1]])
        elif k == "graph":
            edgelist = [t[0] + "\t" + t[1] + "\n" for t in v.edges]
            with open(path / "edge_list.txt", "w") as o:
                o.writelines(edgelist)

    with open(path / "classes.txt", "w") as o:
        o.writelines([x + "\n" for x in classes.keys()])
