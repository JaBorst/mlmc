from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf


def makesequencelabels(query, maxlen, tagset):
    tagsetmap = dict(zip(tagset, range(len(tagset))))
    labels = [[tagsetmap[token] for token in lseq] for lseq in query]
    length = [len(x) for x in query]
    labels = [to_categorical(x, len(tagset)) for x in labels]

    result = np.full((len(query), maxlen, len(tagset)), 0., dtype="float32")
    for i, e in enumerate(labels):
        result[i, :length[i], :] = e

    return np.stack(result)

def makemultilabels(query, maxlen, tagset=None):
    if tagset is not None:
        tagsetmap = dict(zip(tagset, range(len(tagset))))
        labels = [[tagsetmap[token] for token in lseq] for lseq in query]
    else:
        labels = query
    return tf.concat([[tf.reduce_sum(tf.one_hot(x, maxlen), 0) for x in labels]], axis=0)

def to_scheme(tagset, scheme="iobes", outside="O"):
    scheme = scheme.lower()
    if scheme == "iobes" or scheme == "bioes":
        return ([outside] + ["I-" + x for x in tagset] + ["B-" + x for x in tagset] + ["E-" + x for x in tagset] + [
            "S-" + x for x in tagset])
    elif (scheme == "bilou"):
        return (
                [outside] + ["B-" + x for x in tagset] + ["I-" + x for x in tagset] + ["L-" + x for x in tagset] + [
            "U-" + x for x in tagset])
    elif (scheme == "bio" or scheme == "iob"):
        return ([outside] + ["B-" + x for x in tagset] + ["I-" + x for x in tagset])
    elif (scheme == "noprefix"):
        return [outside] + tagset

def schemetransformer(column, scheme="BIOES", multilabel=False):
    scheme = scheme.upper()
    scheme = "BIOES" if scheme == "IOBES" else scheme
    data = column#self.data_manager.get_data(purpose=purpose)
    transforming_column = [tagset if isinstance(tagset, list) else [tagset] for tagset in data]

    newtags_column = []
    for i, tagset in enumerate(transforming_column):
        if "O" in tagset or "O" == tagset:
            newtags_column.append(tagset)
        elif scheme == "NOPREFIX":
            newtags_column.append([t.split("-")[-1] for t in tagset])
        else:
            currentTagset = [t.split("-")[-1] for t in tagset]
            newtags = []
            if i == 0:
                nextTagset = [t.split("-")[-1] for t in transforming_column[i + 1]]
                for t in currentTagset:
                    if t in nextTagset:
                        newtags_column.append("B-" + t)
                    else:
                        if scheme == "BIOES": newtags.append("S-" + t)
                        if scheme == "BIO1": newtags.append("I-" + t)
                        if scheme == "BIO2": newtags.append("B-" + t)
                        newtags_column.append(newtags)
            elif i == len(transforming_column) - 1:
                previousTagset = [t.split("-")[-1] for t in
                                  transforming_column[i - 1]]
                for t in currentTagset:
                    if t in previousTagset:
                        if scheme == "BIOES": newtags.append("E-" + t)
                        if scheme == "BIO1": newtags.append("I-" + t)
                        if scheme == "BIO2": newtags.append("I-" + t)
                    else:
                        if scheme == "BIOES": newtags.append("S-" + t)
                        if scheme == "BIO1": newtags.append("I-" + t)
                        if scheme == "BIO2": newtags.append("B-" + t)
                newtags_column.append(newtags)
            else:

                nextTagset = [t.split("-")[-1] for t in
                              transforming_column[i + 1]]
                previousTagset = [t.split("-")[-1] for t in
                                  transforming_column[i - 1]]
                newtags = []
                for t in currentTagset:
                    if t in previousTagset and t in nextTagset:
                        newtags.append("I-" + t)
                    elif t in previousTagset and t not in nextTagset:
                        if scheme == "BIOES": newtags.append("E-" + t)
                        if scheme == "BIO1": newtags.append("I-" + t)
                        if scheme == "BIO2": newtags.append("I-" + t)
                    elif t not in previousTagset and t in nextTagset:
                        newtags.append("B-" + t)
                    else:
                        if scheme == "BIOES": newtags.append("S-" + t)
                        if scheme == "BIO1": newtags.append("I-" + t)
                        if scheme == "BIO2": newtags.append("B-" + t)
                newtags_column.append(newtags)

    if not multilabel:
        newtags_column = [x[0] for x in newtags_column]
    return newtags_column