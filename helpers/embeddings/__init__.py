try:
    from .embeddings import embed
except:
    print("Flair not installed...")
from .character import charindex
from .labels import makesequencelabels, schemetransformer, to_scheme, makemultilabels