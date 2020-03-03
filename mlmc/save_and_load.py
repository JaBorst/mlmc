from pathlib import Path

import torch
import warnings

from mlmc.representation import is_transformer


def save(model, path, only_inference=False):
    # Remove  loss and optimizer from model, needs to be saved separately
    optimizer_tmp = model.optimizer
    loss_tmp = model.loss

    model.optimizer = None
    model.loss = None

    # # if save_embeddings:
    # #     attr_keys = {x for x in dir(model) if x.startswith("tokenizer") or x.endswith("tokenizer")}
    # # else:
    # #     attr_keys = [x for x in dir(model) if
    # #                  x.startswith("tokenizer") or x.endswith("tokenizer") or x.startswith("embedding") or x.endswith(
    # #                      "embedding")]
    #
    # tmp = {k: getattr(model, k) for k in attr_keys}
    # for k in attr_keys: model.__delattr__(k)

    if only_inference:
        # Use torch.save to save the inference state. if save_all: Save the input representation (embedding or lm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(model, Path(path))

    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            args = type(model).__init__.__code__.co_varnames[1:-1]
            values = {v: model.__dict__[v] for v in args if v != "kwargs"}
            torch.save({
                "type": type(model),
                "args": values,
                "optimizer": optimizer_tmp,
                "loss": loss_tmp,
                "model_state_dict": model.state_dict()}, path)


    # Reattach loss and optimizer and variables
    # for k in attr_keys: setattr(model, k, tmp[k])
    model.loss = loss_tmp
    model.optimizer = optimizer_tmp
    return path


def load(path, only_inference=False):
    if only_inference:
        loaded = torch.load(Path(path))
        return loaded
    else:
        #load all information
        loaded = torch.load(Path(path))
        # Create a model with the same parameters
        model = loaded["type"](**loaded["args"])
        model.load_state_dict(loaded["model_state_dict"])
        model.optimizer = loaded["optimizer"]
        model.loss = loaded["loss"]
        return model
