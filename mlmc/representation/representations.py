import numpy as np
from transformers import *
import torch

MODELS = {"bert": (BertModel, BertTokenizer, 'bert-large-uncased'),
          "bert_cased": (BertModel, BertTokenizer, 'bert-base-cased'),
          "albert": (AlbertModel, AlbertTokenizer, 'albert-large-v2'),
          "gpt": (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
          "gpt2": (GPT2Model, GPT2Tokenizer, 'gpt2'),
          "ctrl": (CTRLModel, CTRLTokenizer, 'ctrl'),
          "xlnet": (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
          "xlm": (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
          "distilbert": (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          "roberta": (RobertaModel, RobertaTokenizer, 'roberta-base'),
          }



def load_static(embedding="/disk1/users/jborst/Data/Embeddings/glove/en/glove.6B.50d_small.txt"):
    glove = np.loadtxt(embedding, dtype='str', comments=None)
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
    ind = [[vocab.get(token, vocab["<UNK_TOKEN>"]) for token in s.split()] for s in query]
    result = torch.zeros((len(query),maxlen))
    for i, e in enumerate(ind):
       result[i,:min(len(e),maxlen)] = torch.Tensor(e[:min(len(e),maxlen)])
    return result


def get_embedding(name, **kwargs):
    weights, vocabulary = load_static(name)
    e = torch.nn.Embedding(weights.shape[0], weights.shape[1],)
    e = e.from_pretrained(torch.Tensor(weights).float(), **kwargs)
    def tokenizer(x, maxlen=500):
        x = [x] if isinstance(x, str) else x
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
        def list_tokenizer(x, maxlen=500):
            l = len(x.split()) if isinstance(x, str) else max([len(s.split()) for s in x])
            x = [x] if isinstance(x, str) else x
            i = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor([tokenizer.encode(sentence, add_special_tokens=False, pad_to_max_length=True)][0]) for sentence in x], batch_first=True)
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