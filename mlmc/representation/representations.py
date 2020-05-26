"""
Loading Embeddings and Word embeddings in an automated fashion.
"""
from pathlib import Path
import shutil
from urllib import error
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import os
import torch
from io import BytesIO
from transformers import *

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
with open(dir_path/"model.txt", "r") as f:
    MODELS = {k.replace("\n", ""): (AutoModel, AutoTokenizer, k.replace("\n","")) for k in f.readlines()}

for k, v in {"bert": (BertModel, BertTokenizer, 'bert-large-uncased'),
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
EMBEDDINGCACHEINDEX = Path.home() / ".mlmc" / "embedding" / "index.txt"

def custom_embedding(name, file):
    """
    Add a custom static embedding file to the cache.

    Use this to register a new embedding file. The file should be a txt file and in the glove format. After loading the file
    with this function you will be able to load the embedding in any model by its name.

    Args:
        name: Name of the embedding for later use
        file: File of the embedding
    Returns:

    """
    source_location = Path(file)
    target_location = EMBEDDINGCACHE / source_location.name
    if (EMBEDDINGCACHE/source_location.name).exists():
        print("Embedding already cached")
    else:
        shutil.copy(source_location, target_location)
        with open(EMBEDDINGCACHEINDEX, "a") as f:
            f.write(name + "\t" + source_location.name)
        print("Cached successfully you can now load [%s] in any model." % (name,))

def delete_custom_embeddings():
    """
    Delete Custom Embeddings from Cache
    Returns:

    """
    with open (EMBEDDINGCACHEINDEX, "r") as f:
        for x in f.read().split("\n"):
            fp = EMBEDDINGCACHE / x.split("\t")[1]
            print(fp)
            fp.unlink()
    EMBEDDINGCACHEINDEX.unlink()

def empty_cache():
    """
    Delete everything from representations cache.

    This does only include static embeddings and custom embeddings. Language Models are downloaded in a different location
    defined by the huggingface library for now.

    ToDo: Use mlmc cache for huggingface to have more control when trying to free disk space.

    Returns:

    """
    for file in  EMBEDDINGCACHE.iterdir():
        file.unlink()

def add_test_example():
    if not EMBEDDINGCACHE.exists():
        EMBEDDINGCACHE.mkdir(parents=True)
    custom_embedding("test", dir_path / "custom_embedding.txt")


def load_static(embedding):
    """
    Load the embedding from a testfile.

    Args:
        embedding: one of [glove50, glove100, glove200, glove300]

    Returns: The embedding matrix and the vocabulary.

    """
    if EMBEDDINGCACHEINDEX.exists():
        with open (EMBEDDINGCACHEINDEX, "r") as f:
            for x in f.read().split("\n"):
                STATICS[x.split("\t")[0]] = x.split("\t")[1]

    if not (EMBEDDINGCACHE / STATICS[embedding]).exists():
        try:
            resp = urlopen("http://nlp.stanford.edu/data/glove.6B.zip")
        except error.HTTPError:
            print(error.HTTPError)
            return None
        assert resp.getcode() == 200, "Download not found Error: (%i)" % (resp.getcode(),)
        print("Downloading glove vectors... This may take a while...")
        zipfile = ZipFile(BytesIO(resp.read()))
        zipfile.extractall(EMBEDDINGCACHE)
    fp = EMBEDDINGCACHE / STATICS[embedding]

    glove = np.loadtxt(fp, dtype='str', comments=None)
    glove = glove[np.unique(glove[:, :1], axis=0, return_index=True)[1]]
    words = glove[:, 0]
    weights = glove[:, 1:].astype('float')
    weights = np.vstack((
        np.array([0] * len(weights[1])),  # the vector for the masking
        weights,
        np.mean(weights, axis=0)),  # the vector for the masking
    )
    words = words.tolist()+["<UNK_TOKEN>"]
    vocabulary = dict(zip(words, range(1, len(words) + 1)))
    return weights, vocabulary


def map_vocab(query, vocab, maxlen):
    """
    Map a query ( a list of lists of tokens ) to indices using the vocab mapping and
    pad (or cut respectively) all to maxlen.
    Args:
        query: a list of lists of tokens
        vocab: A mapping from tokens to indices
        maxlen: Maximum lengths of the lists

    Returns: A torch.Tensor with shape (len(query), maxlen)

    """
    ind = [[vocab.get(token, vocab["<UNK_TOKEN>"]) for token in s] for s in query]
    result = torch.zeros((len(query), maxlen)).long()
    for i, e in enumerate(ind):
       result[i,:min(len(e), maxlen)] = torch.LongTensor(e[:min(len(e), maxlen)])
    return result

def get_white_space_tokenizer(v):
    def tokenizer(x, maxlen=500):
        x = [x] if isinstance(x, str) else x
        x = [s.lower().split() for s in x]
        return map_vocab(x, v, maxlen).long()
    return tokenizer

def get_embedding(name, **kwargs):
    """
    Load a static word embedding from file.
    Args:
        name: File name of the word embedding. (Expects a text file in the glove format)
    Returns: A tuple of embedding and corresponding tokenizer
    """
    import h5py

    file = EMBEDDINGCACHE / (name + ".h5")
    if not (file).exists():
        weights, vocabulary = load_static(name)
        with h5py.File(file, "w") as f:
            f.create_dataset("weights", data=weights)
            f.create_dataset("vocab", data=str(vocabulary))
    else:
        with h5py.File(file, "r")as f:
            weights = f["weights"][()]
            vocabulary = eval(f["vocab"][()])


    e = torch.nn.Embedding(weights.shape[0], weights.shape[1],)
    e = e.from_pretrained(torch.Tensor(weights).float(), **kwargs)

    tokenizer = get_white_space_tokenizer(vocabulary)
    return e, tokenizer


def get_transformer(model="bert", **kwargs):
    """
    Get function for transformer models
    Args:
        model: Model name
        **kwargs: Additional keyword arguments

    Returns:  A tuple of embedding and corresponding tokenizer

    """
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

def get(model, **kwargs):
    """
        Universal get function for text embedding methods and tokenizers.

    Args:
        model: Model name (one of [ glove50, glove100, glove200, glove300] or any of the models on https://huggingface.co/models
        **kwargs:  Additional arguments in case of transformers. for example ``output_hidden_states=True`` for returning hidden states of transformer models.
            For details on the parameters for the specific models see https://huggingface.co


    Returns:
         A tuple of embedding and corresponding tokenizer

    Examples:
        ```
        embedder, tokenizer = get("bert-base-uncased")
        embedding = embedder(tokenizer("A sentence of various words"))
        ```
    The variable ``embedding`` will contain a torch tensor of shape

     (1, sequence_length, embedding_dim)


    """
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
    """
    A check function. True if the ``name`` argument if found to be a valid transformer model name.

    Args:
        name: model name (see get)

    Returns: bool

    """
    return name in MODELS.keys()