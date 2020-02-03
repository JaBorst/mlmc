import torch
from ..representation import get, is_transformer

def save(model, path, only_inference=True):
    if only_inference:
        # Simple Variables
        optimizer_tmp = model.optimizer
        loss_tmp = model.loss

        model.optimizer = None
        model.loss = None

        if is_transformer(model.representation) is not None:
            embedding_tmp, tokenizer_tmp = model.embedding, model.tokenizer
            model.embedding, model.tokenizer = None, None

        torch.save(model, path)

        if is_transformer(model.representation) is not None:
            model.embedding, model.tokenizer = embedding_tmp, tokenizer_tmp

        model.loss = loss_tmp
        model.optimizer = optimizer_tmp
    else:
        raise NotImplemented

def load(path, only_inference=True):
    if only_inference:
        loaded = torch.load(path)
        loaded._init_input_representations()
        return loaded
    else:
        raise NotImplemented