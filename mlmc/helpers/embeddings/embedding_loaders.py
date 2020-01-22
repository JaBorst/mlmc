import numpy as np
from transformers import *
import torch

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


def get_embedding(name):
    weights, vocabulary = load_static(name)
    e = torch.nn.Embedding(weights.shape[0], weights.shape[1])
    e.from_pretrained(torch.Tensor(weights).float(), freeze=True)
    def tokenizer(x, maxlen=500):
        x = [x] if isinstance(x, str) else x
        return map_vocab(x, vocabulary, maxlen).long()
    return e, tokenizer


def get_transformer(model="bert", **kwargs):


    # Transformers has a unified API
    # for 10 transformer architectures and 30 pretrained weights.
    #          Model          | Tokenizer          | Pretrained weights shortcut
    MODELS = {"bert": (BertModel, BertTokenizer, 'bert-base-uncased'),
              "albert": (AlbertModel, AlbertTokenizer, 'albert-base-v2'),
              "gpt": (OpenAIGPTModel, OpenAIGPTTokenizer, 'openai-gpt'),
              "gpt2": (GPT2Model, GPT2Tokenizer, 'gpt2'),
              "ctrl": (CTRLModel, CTRLTokenizer, 'ctrl'),
              "xlnet": (XLNetModel, XLNetTokenizer, 'xlnet-base-cased'),
              "xlm": (XLMModel, XLMTokenizer, 'xlm-mlm-enfr-1024'),
              "distilbert": (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
              "roberta": (RobertaModel, RobertaTokenizer, 'roberta-base'),
              }

    model_class, tokenizer_class, pretrained_weights = MODELS.get(model,(None,None,None))
    if model_class is None:
        print("Model not found...")
        return None
    else:
        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        def list_tokenizer(x, maxlen=500):
            l = len(x.split()) if isinstance(x, str) else max([len(s.split()) for s in x])
            x = [x] if isinstance(x, str) else x
            i = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor([tokenizer.encode(sentence, add_special_tokens=False)][0]) for sentence in x], batch_first=True)
            i = i[:, :min(maxlen, i.shape[-1])]
            return i

        model = model_class.from_pretrained(pretrained_weights, **kwargs)
        return model, list_tokenizer


def get(static=None, transformer=None):
    assert static is None == transformer is None, "Exactly one of the arguments has to be not None"
    if static is not None:
        return get_embedding(static)
    elif transformer is not None:
        return get_transformer(transformer)
