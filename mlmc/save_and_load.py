import torch
from mlmc.representation import get, is_transformer
import warnings
from pathlib import Path

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(model, Path(path))

        if is_transformer(model.representation) is not None:
            model.embedding, model.tokenizer = embedding_tmp, tokenizer_tmp

        model.loss = loss_tmp
        model.optimizer = optimizer_tmp
    else:
        optimizer_tmp = model.optimizer
        loss_tmp = model.loss

        model.optimizer = None
        model.loss = None

        if is_transformer(model.representation) is not None:
            embedding_tmp, tokenizer_tmp = model.embedding, model.tokenizer
            model.embedding, model.tokenizer = None, None


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            args = type(model).__init__.__code__.co_varnames[1:-1]
            values = {v: model.__dict__[v] for v in args}
            torch.save({
                "type": type(model),
                "args": values,
                "model_state_dict": model.state_dict()}, path)

        if is_transformer(model.representation) is not None:
            model.embedding, model.tokenizer = embedding_tmp, tokenizer_tmp

        model.loss = loss_tmp
        model.optimizer = optimizer_tmp
    return path

def load(path, only_inference=True):
    if only_inference:
        loaded = torch.load(Path(path))
        loaded._init_input_representations()
        return loaded
    else:
        loaded  = torch.load(Path(path))
        representation = loaded["args"]["representation"]
        if is_transformer(representation):
            model = loaded["type"](**loaded["args"])
            tmp = model.embedding
            model.embedding=None
            model.load_state_dict(loaded["model_state_dict"])
            model.embedding=tmp
        else:
            model = loaded["type"](**loaded["args"])
            model.load_state_dict(loaded["model_state_dict"])
        return model