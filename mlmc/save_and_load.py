"""
A save and load function for models..
"""
from pathlib import Path

import torch
import warnings


def save(model, path, only_inference=False):
    """Saving a model to disk

    Args:
        model: The model
        path:  The path to save the model to
        only_inference: If False, the optimizer and loss function are saved to disk aswell.
    Returns:
        The path the model was saved to
    """
    # Remove  loss and optimizer from model, needs to be saved separately
    optimizer_tmp = model.optimizer
    loss_tmp = model.loss

    model.optimizer = None
    model.loss = None

    if only_inference:
        # Use torch.save to save the inference state. if save_all: Save the input representation (embedding or lm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(model, Path(path))

    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            args = type(model).__init__.__code__.co_varnames[1:-1] + tuple(set(super(type(model), model).__init__.__code__.co_varnames[1:-1])-set(["optimizer", "loss"]))
            if type(model).__name__ != "LSAN":
                values = {v: model.__dict__[v] for v in args if v != "kwargs"}
            else:
                values = {v: model.__dict__[v] for v in args if v not in ["kwargs", "label_embed"]}
                values["label_embed"] = model.__dict__["label_embed_tmp"]
            torch.save({
                "type": type(model),
                "args": values,
                "optimizer": optimizer_tmp,
                "loss": loss_tmp,
                "model_state_dict": model.state_dict()}, path)


    # Reattach loss and optimizer and variables
    model.loss = loss_tmp
    model.optimizer = optimizer_tmp
    return path


def load(path, only_inference=False, only_cpu=False):
    """
    Load a model from disk

    Args:
        path: The path to load the model from
        only_inference: If the model was saved with only_inference=True this should be True aswell.
        only_cpu: If the model was trained on a GPU, but is trying to be loaded on a system without GPUs set ``only_cpu=True``
            to map the model into RAM instead of GPU.

    Returns:
        The loaded model.

    """
    additional_arguments = {}
    if only_cpu:
        additional_arguments["map_location"] = torch.device('cpu')

    if only_inference:
        loaded = torch.load(Path(path),**additional_arguments)
        return loaded
    else:
        #load all information
        loaded = torch.load(Path(path),**additional_arguments)
        # Create a model with the same parameters
        model = loaded["type"](**loaded["args"])
        model.load_state_dict(loaded["model_state_dict"])
        model.optimizer = loaded["optimizer"]
        model.loss = loaded["loss"]
        return model
