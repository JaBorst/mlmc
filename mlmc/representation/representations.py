"""
Loading Embeddings and Word embeddings in an automated fashion.
"""
import numpy as np
from transformers import *
import torch
from  pathlib import Path
from urllib import error
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+"/model.txt", "r") as f : MODELS = {k.replace("\n",""): (AutoModel, AutoTokenizer,k.replace("\n","")) for k in f.readlines()}

for k,v in {"bert": (BertModel, BertTokenizer, 'bert-large-uncased'),
          "albert": (AlbertModel, AlbertTokenizer, 'albert-large-v2'),
          "ctrl": (CTRLModel, CTRLTokenizer, 'ctrl'),
          "distilbert": (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          "roberta": (RobertaModel, RobertaTokenizer, 'roberta-base'),
          }.items():
    if k not in MODELS.keys():
        MODELS[k]=v

STATICS = {
    "glove50": "glove.6B.50d.txt",
    "glove100": "glove.6B.100d.txt",
    "glove200": "glove.6B.200d.txt",
    "glove300": "glove.6B.300d.txt"
}

EMBEDDINGCACHE = Path.home() / ".mlmc" / "embedding"

def load_static(embedding="glove300"):

    if not (EMBEDDINGCACHE / STATICS[embedding]).exists():
        URL ="http://nlp.stanford.edu/data/glove.6B.zip"
        try:
            resp = urlopen(URL)
        except error.HTTPError:
            print(error.HTTPError)
            return None
        assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)
        print("Downloading glove vectors... This may take a while...")
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall(EMBEDDINGCACHE)
    fp = EMBEDDINGCACHE / STATICS[embedding]

    glove = np.loadtxt(fp, dtype='str', comments=None)
    glove = glove[np.unique(glove[:,:1],axis=0, return_index=True)[1]]
    words = glove[:, 0]
    weights = glove[:, 1:].astype('float')
    weights = np.vstack((
                            np.array([0]* len(weights[1])), # the vector for the masking
                            weights,
                            np.mean(weights, axis=0)), # the vector for the masking)
    )
    words = words.tolist()+["<UNK_TOKEN>"]
    vocabulary = dict(zip(words,range(1,len(words)+1)))
    return weights, vocabulary

def map_vocab(query, vocab, maxlen):
    ind = [[vocab.get(token, vocab["<UNK_TOKEN>"]) for token in s] for s in query]
    result = torch.zeros((len(query),maxlen)).long()
    for i, e in enumerate(ind):
       result[i,:min(len(e),maxlen)] = torch.LongTensor(e[:min(len(e),maxlen)])
    return result


def get_embedding(name, **kwargs):
    weights, vocabulary = load_static(name)
    e = torch.nn.Embedding(weights.shape[0], weights.shape[1],)
    e = e.from_pretrained(torch.Tensor(weights).float(), **kwargs)
    def tokenizer(x, maxlen=500):
        x = [x] if isinstance(x, str) else x
        x = [s.lower().split() for s in x]
        return map_vocab(x, vocabulary, maxlen).long()
    return e, tokenizer


def get_transformer(model="bert", **kwargs):
    # Transformers has a unified API
    # for 10 transformer architectures and 30 pretrained weights.
    #          Model          | Tokenizer          | Pretrained weights shortcut


    model_class, tokenizer_class, pretrained_weights = MODELS.get(model,(None,None,None))
    if model_class is None:
        print("Model is not a transformer...")
        return None
    else:
        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

        def list_tokenizer(x, maxlen=500, return_start=False):
            x = [x] if isinstance(x, str) else x
            if return_start:
                i = [tokenizer.tokenize(sentence, add_special_tokens=False, pad_to_max_length=True) for sentence in x]
                ind = [torch.tensor([x.startswith("Ġ") or i == 0 for i, x in enumerate(sentence)]) for sentence in i]
                i = torch.nn.utils.rnn.pad_sequence([torch.tensor(tokenizer.convert_tokens_to_ids(a)) for a in i],
                                                    batch_first=True, padding_value=tokenizer.pad_token_id)
                return i, torch.nn.utils.rnn.pad_sequence(ind, batch_first=True, padding_value=False)
            else:
                i = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor([tokenizer.encode(sentence, add_special_tokens=False, pad_to_max_length=True)][0]) for
                     sentence in x], batch_first=True, padding_value=tokenizer.pad_token_id)
            i = i[:, :min(maxlen, i.shape[-1])]
            return i

        model = model_class.from_pretrained(pretrained_weights, **kwargs)
        return model, list_tokenizer


def get_by_arg_(static=None, transformer=None, **kwargs):
    assert (static is None) != (transformer is None), "Exactly one of the arguments has to be not None"
    if static is not None:
        return get_embedding(static, **kwargs)
    elif transformer is not None:
        import logging
        print("Setting transformers.tokenization_utils logger to ERROR.")
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
        return get_transformer(transformer, **kwargs)

def get(model, **kwargs):
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    module = get_transformer(model, **kwargs)
    if module is None:
        module = get_embedding(model, **kwargs)
        if module is None:
            raise FileNotFoundError
        return module
    else:
        return module

def is_transformer(name):
    return name in MODELS.keys()