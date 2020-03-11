"""
Functions and classes for loading and using numerical text representations like word embeddings
and language models in an automated fashion.
"""

from .representations import load_static, get_transformer, map_vocab,get_embedding,get, is_transformer
from .character import charindex
from .labels import makesequencelabels, schemetransformer, to_scheme, makemultilabels
from .postprocessing_vectors import postprocess_embedding
from .label_embeddings import get_word_embedding_mean, get_lm_generated, get_lm_repeated
from .embedder import Embedder