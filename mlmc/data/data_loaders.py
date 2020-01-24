import os
import json
from tqdm import tqdm
import numpy as np
import networkx as nx
###
# Repository http://manikvarma.org/downloads/XC/XMLRepository.html
#
#
#
#
#
#
tmp_dir = os.path.join(os.getenv("HOME"),".mlmc/datasets/")


def _load_from_tmp(dataset):
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    if os.path.isfile(os.path.join(tmp_dir,dataset)):
        import pickle
        print("Loading from cache...")
        with open(os.path.join(tmp_dir, dataset), "rb") as f: data = pickle.load(f)
        return data
    else:
        return None

def _save_to_tmp(dataset, data):
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    if not os.path.isfile(os.path.join(tmp_dir,dataset)):
        import pickle
        with open(os.path.join(tmp_dir, dataset), "wb") as f: pickle.dump(data, f)
        return True
    else:
        return False

#============================================================================
#============================================================================

def load_appd(path="/disk1/users/jborst/Data/Test/MultiLabel/AAPD/"):

    dataset=os.path.abspath(path)
    data = _load_from_tmp("appd")
    if data is not None: return data
    else:
        with open(os.path.join(dataset,"aapd.json"))as f: classes = json.load(f)
        data = {}
        for x in ("test","val", "train"):
            with open(os.path.join(dataset,"text_"+x)) as f:test_x = f.readlines()
            with open(os.path.join(dataset,"label_"+x)) as f:
                test_y = [[y.lower() for y in x.replace("\n","").split(" ")] for x in f.readlines()]
            if x == "val":
                data["valid"] = [test_x,test_y]
            else:
                data[x] =[test_x,test_y]
        _save_to_tmp("appd", (data, classes))
    return data, classes



def load_rcv1(path="/disk1/users/jborst/Data/Test/MultiLabel/reuters/corpus-reuters-corpus-vol1/"):
    import zipfile
    from xml.etree import ElementTree
    import tempfile

    dataset=os.path.abspath(path)
    data = _load_from_tmp("rcv1")
    if data is not None: return data

    with open(os.path.join(dataset,"categories.txt"))as f:
        labels={}
        needed_zips = set()
        for x in f.readlines():
            if x.split()[0][-4:] == ".zip" and len(x.replace("\n", "").split()[2:]) > 0:
                needed_zips.add(x.split()[0])
                labels[x.split()[1]] = x.replace("\n","").split()[2:]

    with open(os.path.join(dataset, "train.split"))as f: train_ids = [x.replace("\n","") for x in f.readlines()]
    with open(os.path.join(dataset, "test.split"))as f: test_ids = [x.replace("\n","") for x in f.readlines()]

    with open(os.path.join(dataset,"topic_codes.txt"))as f:
        classes = [(x.split("\t")[0], x.replace("\n","").split("\t")[1:]) for x in f.readlines()[2:-1]]
        classes = set([x[0] for x in classes])

    with tempfile.TemporaryDirectory() as tempdir:
        for file in tqdm(needed_zips):
            zipfile.ZipFile(os.path.join(dataset,file)).extractall(tempdir)

        def _get(ids):
            errors=0
            documents = []
            labelsets = []
            for id in tqdm(ids):
                file = id+"newsML.xml"
                try:
                    with open(os.path.join(tempdir, file)) as f:
                        xml = ElementTree.fromstring(f.read())
                        text = ElementTree.tostring(xml, method='text').decode().replace("\n"," ").replace("  "," ").strip()
                        documents.append(text)
                        labelsets.append([x for x in labels[file] if x in classes])
                except Exception as e:
                    try:
                        with open(os.path.join(tempdir, file), encoding="iso-8859-1") as f:
                            xml = ElementTree.fromstring(f.read())
                            text = ElementTree.tostring(xml, method='text').decode().replace("\n"," ").replace("  "," ").strip()
                            documents.append(text)
                            labelsets.append([x for x in labels[file] if x in classes])
                    except Exception as e2:
                        errors +=1
                        print(e2)
            print(str(errors) + " Errors")
            return (documents, labelsets)

        train = _get(train_ids)
        test = _get(test_ids)

    classes = list(classes.intersection(set([x for  y in train[1]+test[1] for x in y])))
    classes = dict(zip(classes, range(len(classes))))
    data = {}
    data["train"] = train
    data["test"] = test
    with open(os.path.join(path, "rcv1.topics.hier.orig"), "r") as f:
        content = f.readlines()
    import re
    edges = [(re.split(" +", x)[1],re.split(" +", x)[3]) for x in content]
    graph = nx.DiGraph(edges)
    data["graph"] = graph
    with open(os.path.join(path,"topic_codes.txt"), "r") as f:
        topics = [x.replace("\n", "").split("\t") for x in f.readlines() if len(x) > 1][2:]
    topicmap = {x[0]: x[1] for x in topics}
    data["topicmap"]=topicmap

    _save_to_tmp("rcv1", (data, classes))
    return data, classes


def load_wiki30k(path="/disk1/users/jborst/Data/Test/MultiLabel/wiki30k"):
    import pickle
    with open(os.path.join(path, "wiki30k_raw_text.p"), "rb") as f: content = pickle.load(f)
    train_x = [x["text"] for x in content[0]]
    train_y = [x["catgy"] for x in content[0]]
    test_x = [x["text"] for x in content[1]]
    test_y = [x["catgy"] for x in content[1]]
    data = {}
    data["train"] = (train_x, train_y)
    data["test"] = (test_x, test_y)
    return data, content[3]

def load_eurlex(path="/disk1/users/jborst/Data/Test/MultiLabel/EURLex"):
    import pickle
    with open(os.path.join(path, "eurlex_raw_text.p"), "rb") as f: content = pickle.load(f)
    train_x = [x["text"] for x in content[0]]
    train_y = [x["catgy"] for x in content[0]]
    test_x = [x["text"] for x in content[1]]
    test_y = [x["catgy"] for x in content[1]]
    classes = content[3]

    reverse_classes= {v:k for k,v in classes.items()}
    train_y = [[reverse_classes[x] for x in labels]  for labels in train_y]
    test_y = [[reverse_classes[x] for x in labels]  for labels in test_y]
    data = {}
    data["train"] = (train_x, train_y)
    data["test"] = (test_x, test_y)
    return data, classes

def load_huffpost(path="/disk1/users/jborst/Data/Test/MultiLabel/HuffPost", test_split=0.25):
    import json
    from sklearn.model_selection import train_test_split

    data = _load_from_tmp("huffpost")
    if data is not None: return data
    else:

        with open(os.path.join(path, "News_Category_Dataset_v2.json"), "r") as f: content = json.loads("["+",".join(f.readlines())+"]")
        headlines = [x["headline"] for x in content]
        label = [x["category"] for x in content]
        data = (headlines, label)
        classes = list(set(label))
        classes = dict(zip(classes, range(len(classes))))
        n_arr = len(data)
        data = train_test_split(*data, test_size=test_split)
        tmp = [[],[]]
        for i, arr in enumerate(data):
            tmp[i % n_arr].append(arr)

        _save_to_tmp("huffpost", (data, classes))

    return {"train": tmp[0], "test": tmp[1]}, classes

def load_moviesummaries(path="/disk1/users/jborst/Data/Test/MultiLabel/MovieSummaries", test_split=0.25):
    data = _load_from_tmp("moviesummaries")
    if data is not None: return data
    else:

        from sklearn.model_selection import train_test_split
        with open(os.path.join(path, "plot_summaries.txt"), "r") as f: content = f.readlines()
        with open(os.path.join(path, "movie.metadata.tsv"), "r") as f: meta = {x.split("\t")[0]:x.split("\t")[-1].replace("\n","") for x in f.readlines()}#f.readlines()#[x.split("\t")[-1].replace("\n","") for x in f.readlines()]
        meta = {k:[x.split(": ")[-1].replace("\"","").replace("}","") for x in genre.split(", ")] for k,genre in meta.items()}

        data = [(x.split("\t")[1], meta[str(x.split("\t")[0])]) for x in content if str(x.split("\t")[0]) in meta.keys()]
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

        _save_to_tmp("huffpost", (data, classes))

        return {"train": tmp[0], "test": tmp[1]}, classes




################################
#### Hierarchical Multilabel
#### see http://kt.ijs.si/DragiKocev/PhD/resources/doku.php?id=hmc_classification
################################

def load_blurbgenrecollection():
    url = "https://fiona.uni-hamburg.de/ca89b3cf/blurbgenrecollectionen.zip"
    data = _load_from_tmp("blurbgenrecollection")
    if data is not None: return data
    else:
        from bs4 import BeautifulSoup
        from xml.etree import ElementTree
        from urllib.request import urlopen
        from zipfile import ZipFile
        from io import BytesIO
        import re

        resp = urlopen(url)
        zipfile = ZipFile(BytesIO(resp.read()))

        data = {}
        for purpose in ["dev", "train", "test"]:
            soup = BeautifulSoup("<root>"+ zipfile.open("BlurbGenreCollection_EN_"+purpose+".txt").read().decode("utf-8").replace("\n","") + "</root>")
            text, labels = [], []
            for i in soup.findAll("book"):
                text.append(i.find("body").text)
                labels.append([x.text for x in i.find("topics").children])
            if purpose=="dev":
                data["valid"]=(text,labels)
            else:
                data[purpose]=(text,labels)

        edges = [x.decode("utf-8").replace("\n","").split("\t") for x in zipfile.open("hierarchy.txt").readlines()]
        classes = list(set([x for y in data["train"][1] +data["valid"][1] +data["test"][1]+edges for x in y]))
        classes = dict(zip(classes, range(len(classes))))
        edges = [[classes[x] for x in e] for e in edges if len(e)==2]
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
        from bs4 import BeautifulSoup
        from xml.etree import ElementTree
        from urllib.request import urlopen
        from zipfile import ZipFile
        from io import BytesIO
        import re

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
        edges = [[classes[x] for x in e] for e in edges if len(e) == 2]
        graph = nx.DiGraph(edges)
        data["graph"] = graph
        _save_to_tmp("blurbgenrecollection_de", (data, classes))
        return data, classes

def load_webofscience():
    url = "https://data.mendeley.com/datasets/9rw3vkcfy4/6/files/c9ea673d-5542-44c0-ab7b-f1311f7d61df/WebOfScience.zip?dl=1"
    data = _load_from_tmp("blurbgenrecollection")
    if data is not None:
        return data
    else:
        from xml.etree import ElementTree
        from urllib.request import urlopen
        from zipfile import ZipFile
        from io import BytesIO
        import re

        resp = urlopen(url)
        zipfile = ZipFile(BytesIO(resp.read()))

################################
# Named Entity Recognition
################################
import re

def read_conll(file, column=3):
    """Read in the standard tab separated conll format"""
    with open(file, "r", encoding="utf-8") as file_handler:
        content = file_handler.read().strip("\n").strip()
        sentences = [[x.split(" ")[0].strip(" ").strip("\t") for x in sentence.split("\n")] for sentence in re.split(r"\n\s*\n*\s*\n", content)]
        ner = [[x.split(" ")[column].strip(" ").strip("\t") for x in sentence.split("\n")] for sentence in re.split(r"\n\s*\n*\s*\n", content)]
    return sentences, ner


def load_conll2003en(path="/disk1/users/jborst/Data/Test/NER/CoNLL-2003/eng/BIOES/"):
    data = _load_from_tmp("conll2003en")
    if data is not None: return data
    else:
        data = {}
        data["test"]= read_conll(os.path.join(path,"test.txt"))
        data["valid"]= read_conll(os.path.join(path,"valid.txt"))
        data["train"]= read_conll(os.path.join(path,"train.txt"))
        from mlmc.data.tagsets import NER
        classes = dict(zip(NER, range(len(NER))))
        _save_to_tmp("conll2003en", (data, classes))
    return data, classes

def load_20newsgroup():
    url = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
    data = _load_from_tmp("20newsgroup")
    if data is not None:
        return data
    else:
        from urllib.request import urlopen
        import tarfile
        from io import BytesIO
        import tempfile
        import os
        tmpdir = tempfile.TemporaryDirectory()

        resp = urlopen(url)
        tf = tarfile.open(fileobj=resp, mode="r|gz")
        tf.extractall(tmpdir.name)
        testdir = os.path.join(tmpdir.name,'20news-bydate-test')
        traindir = os.path.join(tmpdir.name,'20news-bydate-train')

        classes = []

        testdata = {"text": [], "label": []}
        for catg in os.listdir(testdir):
            classes.append(catg)
            for file in os.listdir(os.path.join(testdir, catg)) :
                with open(os.path.join(testdir, catg, file), 'r', encoding="ISO-8859-1") as f:
                    testdata["text"].append(f.read())
                    testdata["label"].append([catg])

        traindata = {"text": [], "label": []}
        for catg in os.listdir(traindir):
            classes.append(catg)
            for file in os.listdir(os.path.join(traindir, catg)):
                with open(os.path.join(traindir, catg, file), 'r', encoding="ISO-8859-1") as f:
                    traindata["text"].append(f.read())
                    traindata["label"].append([catg])


        classes = list(set(classes))
        classes.sort()
        edges = [
            (x.split(".")[y], x.split(".")[y+1] if y+2 != len(x.split(".")) else x)
            for x in classes for y in range(len(x.split("."))-1) ]
        edges += list(set([("ROOT", x.split(".")[0]) for x in classes]))

        graph =  nx.DiGraph(edges)
        _save_to_tmp("20newsgroup", (data, classes))

        return {"train": traindata, "test": testdata, "graph": graph}, classes
