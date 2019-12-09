import os
import json
from tqdm import tqdm
import numpy as np
tmp_dir = os.path.join(os.getenv("HOME"),".mlmc/datasets/")


def load_from_tmp(dataset):
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    if os.path.isfile(os.path.join(tmp_dir,dataset)):
        import pickle
        print("Loading from cache...")
        with open(os.path.join(tmp_dir, dataset), "rb") as f: data = pickle.load(f)
        return data
    else:
        return None

def save_to_tmp(dataset, data):
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
    data = load_from_tmp("appd")
    if data is not None: return data
    else:
        with open(os.path.join(dataset,"aapd.json"))as f: classes = json.load(f)
        data = {}
        for x in ("test","val", "train"):
            with open(os.path.join(dataset,"text_"+x)) as f:test_x = f.readlines()
            with open(os.path.join(dataset,"label_"+x)) as f:
                test_y = [[y.lower() for y in x.replace("\n","").split(" ")] for x in f.readlines()]
            data[x] =[test_x,test_y]
        save_to_tmp("appd",(data,classes))
    return data, classes

import zipfile
from xml.etree import ElementTree
import tempfile
def load_rcv1(path="/disk1/users/jborst/Data/Test/MultiLabel/reuters/corpus-reuters-corpus-vol1/"):
    dataset=os.path.abspath(path)
    data = load_from_tmp("rcv1")
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
    save_to_tmp("rcv1", (data,classes))
    return data, classes


def load_amazon12k(vocabulary, path="/disk1/users/jborst/Data/Test/MultiLabel/Amazon12k/", model=None):
    import pickle
    with open(os.path.join(path, "amazon12k_raw_text.p"), "rb") as f: content = pickle.load(f)


def load_wiki30k( path="/disk1/users/jborst/Data/Test/MultiLabel/wiki30k"):
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

def load_eurlex(vocabulary, path="/disk1/users/jborst/Data/Test/MultiLabel/EURLex", model=None):
    import pickle
    with open(os.path.join(path, "eurlex_raw_text.p"), "rb") as f: content = pickle.load(f)
    train_x = [x["text"] for x in content[0]]
    train_y = [x["catgy"] for x in content[0]]
    test_x = [x["text"] for x in content[1]]
    test_y = [x["catgy"] for x in content[1]]
    classes = content[3]
    data = {}
    if model is None:
        data["train"] = (train_x, train_y)
        data["test"] = (test_x, test_y)
    return {"train": content[0], "test": content[1]}, classes

def load_huffpost(path="/disk1/users/jborst/Data/Test/MultiLabel/HuffPost", test_split=0.3):
    import json
    from sklearn.model_selection import train_test_split

    data = load_from_tmp("huffpost")
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

        save_to_tmp("huffpost", (data,classes))

    return {"train": tmp[0], "test": tmp[1]}, classes

