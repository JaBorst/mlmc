"""
Loading Embeddings and Word embeddings in an automated fashion.
"""
from pathlib import Path
import shutil
from urllib import error
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import tempfile
import torch
from io import BytesIO
from transformers import AutoModel, AutoTokenizer
import pathlib
import logging


EMBEDDINGCACHE = Path.home() / ".mlmc" / "embedding"
EMBEDDINGCACHEINDEX = Path.home() / ".mlmc" / "embedding" / "index.txt"

if not EMBEDDINGCACHE.exists():
    EMBEDDINGCACHE.mkdir(parents=True)
if not EMBEDDINGCACHEINDEX.exists():
    EMBEDDINGCACHEINDEX.touch()

def get_transformer(model="bert", **kwargs):
    """
    Get function for transformer models
    Args:
        model: Model name
        **kwargs: Additional keyword arguments

    Returns:  A tuple of embedding and corresponding tokenizer

    """
    model_class, tokenizer_class, pretrained_weights = AutoModel, AutoTokenizer, model

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights, **kwargs)
    return model, tokenizer


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
    try:
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    except:
        logging.get_logger("transformers.tokenization_utils").setLevel(logging.ERROR)

    module = get_transformer(model)
    return module
