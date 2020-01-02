from .embedding_loaders import load_glove
from .character import charindex
from .labels import makesequencelabels, schemetransformer, to_scheme, makemultilabels
from .embeddings import embed, get_embedder
# try:
#     from .embeddings import embed
# except:
#     print("Flair not installed...")
